//! Composition for [`Element`]s using drm planes
//!
//! When possible composition can be (partially) offloaded to the display driver by assigning
//! elements to drm planes. This is especially important for latency intensive fullscreen clients
//! like video renderers or games.
//!
//! The [`DrmCompositor`] does so by walking the stack of provided [`Element`]s from front to back
//! while trying to assign each element to a drm overlay plane. Each item that fails the plane test
//! will be rendered on the primary plane using the provided [`Renderer`].
//! Additionally it will try to assign the top most element that fit's into the cursor size (as specified
//! by the [`DrmDevice`](crate::backend::drm::DrmDevice)) on the cursor plane. If the element can not be
//! directly scanned out, pixman will be used to render the element.
//!
//! Note: While the [`DrmCompositor`] also works on *legacy* drm the use of overlay and cursor planes is disabled in that case.
//! Direct scan-out will only work with an atomic [`DrmSurface`].
//!
//! ## What makes a [`Element`] eligible for direct scan-out
//!
//! ### General
//!
//! First the element has to provide a [`UnderlyingStorage`] which can be exported as a drm framebuffer.
//! Currently this is limited to wayland buffers, but may be extended in the future.
//! This module provides a default exporter based on [`gbm`] which should fit most use-cases.
//!
//! If a certain combination of elements works can only be determined by asking the driver by submitting
//! a atomic commit test. If that test fails the element is scheduled to be rendered on the primary plane.
//!
//! ### Overlay planes
//!
//! The element can only be directly scanned out if it's geometry does not overlap with an already assigned
//! element on a plane higher in the stack.
//!
//! ### Underlay planes
//!
//! An underlay plane is only used if it does not overlap with an already assigned plane lower in the stack
//! and the element is fully opaque.
//!
//! ### Primary plane
//!
//! For an element to be considered to be directly scanned out on the primary plane it has to be the last remaining
//! visible element on the output and no other element has been assigned to the primary plane. If there are multiple
//! element assigned to the primary plane the renderer will be used to composite the primary plane into a allocator
//! provided buffer. Additionally the element has to be either fully opaque or the clear color has to match the CRTC
//! background color and no overlap with an underlay is found.
//!
//! # How to use it
//!
//! ```no_run
//! # use smithay::backend::{
//! #     allocator::gbm::{GbmAllocator, GbmDevice},
//! #     drm::{DrmDevice, DrmDeviceFd},
//! #     renderer::{
//! #       element::surface::WaylandSurfaceRenderElement,
//! #       gles::{GlesTexture, GlesRenderer},
//! #     },
//! # };
//! # use drm_fourcc::{DrmFormat, DrmFourcc, DrmModifier};
//! # use std::{collections::HashSet, mem::MaybeUninit};
//! #
//! use smithay::{
//!     backend::drm::{
//!         compositor::{DrmCompositor, FrameFlags},
//!         exporter::gbm::GbmFramebufferExporter,
//!         DrmSurface,
//!     },
//!     output::{Output, PhysicalProperties, Subpixel},
//!     utils::Size,
//! };
//!
//! // ...initialize the output, drm device, drm surface and allocator
//! #
//! # const CLEAR_COLOR: [f32; 4] = [0f32, 0f32, 0f32, 0f32];
//! #
//! let output = Output::new(
//!     "e-DP".into(),
//!     PhysicalProperties {
//!         size: Size::from((800, 600)),
//!         make: "N/A".into(),
//!         model: "N/A".into(),
//!         subpixel: Subpixel::Unknown,
//!         serial_number: "N/A".into(),
//!     },
//! );
//!
//! # let device: DrmDevice = todo!();
//! # let surface: DrmSurface = todo!();
//! # let allocator: GbmAllocator<DrmDeviceFd> = todo!();
//! # let exporter: GbmFramebufferExporter<DrmDeviceFd> = todo!();
//! # let color_formats = [DrmFourcc::Argb8888];
//! # let renderer_formats = HashSet::from([DrmFormat {
//! #     code: DrmFourcc::Argb8888,
//! #     modifier: DrmModifier::Linear,
//! # }]);
//! # let gbm: GbmDevice<DrmDeviceFd> = todo!();
//! # let mut renderer: GlesRenderer = todo!();
//! #
//! let mut compositor: DrmCompositor<_, _, (), _> = DrmCompositor::new(
//!     &output,
//!     surface,
//!     None,
//!     allocator,
//!     exporter,
//!     color_formats,
//!     renderer_formats,
//!     device.cursor_size(),
//!     Some(gbm),
//! )
//! .expect("failed to initialize drm compositor");
//!
//! # let elements: Vec<WaylandSurfaceRenderElement<GlesRenderer>> = Vec::new();
//! let render_frame_result = compositor
//!     .render_frame::<_, _>(&mut renderer, &elements, CLEAR_COLOR, FrameFlags::DEFAULT)
//!     .expect("failed to render frame");
//!
//! if !render_frame_result.is_empty {
//!     compositor.queue_frame(()).expect("failed to queue frame");
//!
//!     // ...wait for VBlank event
//!
//!     compositor
//!         .frame_submitted()
//!         .expect("failed to mark frame as submitted");
//! } else {
//!     // ...re-schedule frame
//! }
//! ```
use std::{
    collections::HashMap,
    fmt::Debug,
    io::ErrorKind,
    os::unix::io::{AsFd, OwnedFd},
    str::FromStr,
    sync::Arc,
};

use drm::{
    Device, DriverCapability,
    control::{Device as _, Mode, PlaneType, connector, crtc, framebuffer, plane},
};
use drm_fourcc::{DrmFormat, DrmFourcc, DrmModifier};
use indexmap::{IndexMap, IndexSet};
use smallvec::SmallVec;
use tracing::{debug, error, info, info_span, instrument, trace, warn};
use wayland_server::{Resource, protocol::wl_buffer::WlBuffer};

#[cfg(feature = "renderer_pixman")]
use crate::backend::renderer::{
    Frame as _, ImportAll,
    pixman::{PixmanError, PixmanRenderer, PixmanTexture},
};
use crate::{
    backend::{
        SwapBuffersError,
        allocator::{
            Allocator, Buffer, Slot, Swapchain,
            dmabuf::{AsDmabuf, Dmabuf},
            format::{get_opaque, has_alpha},
            gbm::{GbmAllocator, GbmBuffer, GbmBufferFlags, GbmDevice},
        },
        drm::{DrmError, PlaneDamageClips, plane_has_property},
        renderer::{
            Bind, Color32F, DebugFlags, PresentationMode, Renderer, RendererSuper, Texture,
            buffer_y_inverted,
            damage::{Error as OutputDamageTrackerError, OutputDamageTracker},
            element::{
                Element, Id, Kind, RenderElement, RenderElementPresentationState, RenderElementState,
                RenderElementStates, RenderingReason, UnderlyingStorage,
            },
            sync::SyncPoint,
            utils::{CommitCounter, DamageBag},
        },
    },
    output::OutputModeSource,
    utils::{Buffer as BufferCoords, DevPath, Physical, Point, Rectangle, Scale, Size, Transform},
    wayland::{shm, single_pixel_buffer},
};

use super::{
    DrmSurface, Framebuffer, PlaneClaim, PlaneInfo, Planes,
    error::AccessError,
    exporter::{ExportBuffer, ExportFramebuffer, gbm::GbmFramebufferExporter, gbm::NodeFilter},
    surface::VrrSupport,
};

mod elements;
mod frame_result;

use elements::*;
pub use frame_result::*;

impl RenderElementState {
    pub(crate) fn zero_copy(visible_area: usize) -> Self {
        RenderElementState {
            visible_area,
            presentation_state: RenderElementPresentationState::ZeroCopy,
            needs_capture: false,
        }
    }

    pub(crate) fn rendering_with_reason(reason: RenderingReason) -> Self {
        RenderElementState {
            visible_area: 0,
            presentation_state: RenderElementPresentationState::Rendering { reason: Some(reason) },
            needs_capture: false,
        }
    }
}

#[allow(dead_code)] // This structs purpose is to keep buffer objects alive, most variants won't be read
#[derive(Debug)]
enum ScanoutBuffer<B: Buffer> {
    Wayland(crate::backend::renderer::utils::Buffer),
    Swapchain(Arc<Slot<B>>),
    Cursor(Arc<GbmBuffer>),
}

impl<B: Buffer> Clone for ScanoutBuffer<B> {
    fn clone(&self) -> Self {
        match self {
            Self::Wayland(arg0) => Self::Wayland(arg0.clone()),
            Self::Swapchain(arg0) => Self::Swapchain(arg0.clone()),
            Self::Cursor(arg0) => Self::Cursor(arg0.clone()),
        }
    }
}

impl<B: Buffer> ScanoutBuffer<B> {
    fn acquire_point(
        &self,
        signaled_fence: Option<&Arc<OwnedFd>>,
    ) -> Option<(SyncPoint, Option<Arc<OwnedFd>>)> {
        if let Self::Wayland(buffer) = self {
            // Assume `DrmSyncobjBlocker` is used, so acquire point has already
            // been signaled. Instead of converting with `SyncPoint::from`.
            if buffer.acquire_point().is_some() {
                return Some((SyncPoint::signaled(), signaled_fence.cloned()));
            }
        }
        None
    }
}

impl<B: Buffer> ScanoutBuffer<B> {
    #[inline]
    fn from_underlying_storage(storage: UnderlyingStorage<'_>) -> Option<Self> {
        match storage {
            UnderlyingStorage::Wayland(buffer) => Some(Self::Wayland(buffer.clone())),
            UnderlyingStorage::Memory { .. } => None,
        }
    }
}

enum DrmFramebuffer<F: Framebuffer> {
    Exporter(F),
    Gbm(super::gbm::GbmFramebuffer),
}

impl<F> AsRef<framebuffer::Handle> for DrmFramebuffer<F>
where
    F: Framebuffer,
{
    #[inline]
    fn as_ref(&self) -> &framebuffer::Handle {
        match self {
            DrmFramebuffer::Exporter(e) => e.as_ref(),
            DrmFramebuffer::Gbm(g) => g.as_ref(),
        }
    }
}

impl<F> Framebuffer for DrmFramebuffer<F>
where
    F: Framebuffer,
{
    #[inline]
    fn format(&self) -> drm_fourcc::DrmFormat {
        match self {
            DrmFramebuffer::Exporter(e) => e.format(),
            DrmFramebuffer::Gbm(g) => g.format(),
        }
    }
}

impl<F> std::fmt::Debug for DrmFramebuffer<F>
where
    F: Framebuffer + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exporter(arg0) => f.debug_tuple("Exporter").field(arg0).finish(),
            Self::Gbm(arg0) => f.debug_tuple("Gbm").field(arg0).finish(),
        }
    }
}

struct DrmScanoutBuffer<B: Buffer, F: Framebuffer> {
    buffer: ScanoutBuffer<B>,
    fb: CachedDrmFramebuffer<F>,
}

impl<B: Buffer, F: Framebuffer> Clone for DrmScanoutBuffer<B, F> {
    fn clone(&self) -> Self {
        DrmScanoutBuffer {
            buffer: self.buffer.clone(),
            fb: self.fb.clone(),
        }
    }
}

impl<B, F> std::fmt::Debug for DrmScanoutBuffer<B, F>
where
    B: Buffer + std::fmt::Debug,
    F: Framebuffer + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DrmScanoutBuffer")
            .field("buffer", &self.buffer)
            .field("fb", &self.fb)
            .finish()
    }
}

impl<B: Buffer, F: Framebuffer> AsRef<framebuffer::Handle> for DrmScanoutBuffer<B, F> {
    #[inline]
    fn as_ref(&self) -> &drm::control::framebuffer::Handle {
        self.fb.as_ref()
    }
}

impl<B: Buffer, F: Framebuffer> Framebuffer for DrmScanoutBuffer<B, F> {
    #[inline]
    fn format(&self) -> drm_fourcc::DrmFormat {
        self.fb.format()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
enum ElementFramebufferCacheBuffer {
    Wayland(wayland_server::Weak<WlBuffer>),
}

impl ElementFramebufferCacheBuffer {
    #[inline]
    fn from_underlying_storage(storage: &UnderlyingStorage<'_>) -> Option<Self> {
        match storage {
            UnderlyingStorage::Wayland(buffer) => Some(Self::Wayland(buffer.downgrade())),
            UnderlyingStorage::Memory { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ElementFramebufferCacheKey {
    allow_opaque_fallback: bool,
    buffer: ElementFramebufferCacheBuffer,
}

impl ElementFramebufferCacheKey {
    #[inline]
    fn from_underlying_storage(storage: &UnderlyingStorage<'_>, allow_opaque_fallback: bool) -> Option<Self> {
        let buffer = ElementFramebufferCacheBuffer::from_underlying_storage(storage)?;
        Some(Self {
            allow_opaque_fallback,
            buffer,
        })
    }
}

impl ElementFramebufferCacheKey {
    #[inline]
    fn is_alive(&self) -> bool {
        match self.buffer {
            ElementFramebufferCacheBuffer::Wayland(ref buffer) => buffer.is_alive(),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct PlanesSnapshot {
    primary: bool,
    cursor_bitmask: u32,
    overlay_bitmask: u32,
}

#[derive(Debug)]
struct ElementInstanceState {
    properties: PlaneProperties,
    active_planes: PlanesSnapshot,
    failed_planes: PlanesSnapshot,
}

#[derive(Debug)]
struct ElementState<B: Framebuffer> {
    instances: SmallVec<[ElementInstanceState; 1]>,
    fb_cache: ElementFramebufferCache<B>,
}

#[derive(Debug)]
struct ElementFramebufferCache<B>
where
    B: Framebuffer,
{
    /// Cache for framebuffer handles per cache key (e.g. wayland buffer)
    fb_cache: SmallVec<
        [(
            ElementFramebufferCacheKey,
            Result<CachedDrmFramebuffer<B>, ExportBufferError>,
        ); 4],
    >,
}

impl<B> ElementFramebufferCache<B>
where
    B: Framebuffer,
{
    #[inline]
    fn get(
        &self,
        cache_key: &ElementFramebufferCacheKey,
    ) -> Option<Result<&CachedDrmFramebuffer<B>, ExportBufferError>> {
        self.fb_cache.iter().find_map(|(k, r)| {
            if k == cache_key {
                Some(r.as_ref().map_err(|err| *err))
            } else {
                None
            }
        })
    }

    #[inline]
    fn insert(
        &mut self,
        cache_key: ElementFramebufferCacheKey,
        fb: Result<CachedDrmFramebuffer<B>, ExportBufferError>,
    ) {
        self.fb_cache.push((cache_key, fb));
    }

    fn cleanup(&mut self) {
        self.fb_cache.retain(|(key, _)| key.is_alive());
    }
}

impl<B> Default for ElementFramebufferCache<B>
where
    B: Framebuffer,
{
    #[inline]
    fn default() -> Self {
        Self {
            fb_cache: Default::default(),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct PlaneProperties {
    pub src: Rectangle<f64, BufferCoords>,
    pub dst: Rectangle<i32, Physical>,
    pub transform: Transform,
    pub alpha: f32,
    pub format: DrmFormat,
}

impl PlaneProperties {
    #[inline]
    fn is_compatible(&self, other: &PlaneProperties) -> bool {
        self.src == other.src
            && self.dst == other.dst
            && self.transform == other.transform
            && self.alpha == other.alpha
            && self.format == other.format
    }
}

struct ElementPlaneConfig<'a, B: Buffer, F: Framebuffer> {
    z_index: usize,
    geometry: Rectangle<i32, Physical>,
    properties: PlaneProperties,
    buffer: DrmScanoutBuffer<B, F>,
    failed_planes: &'a mut PlanesSnapshot,
}

#[derive(Debug)]
struct PlaneConfig<B: Buffer, F: Framebuffer> {
    pub properties: PlaneProperties,
    pub buffer: DrmScanoutBuffer<B, F>,
    pub damage_clips: Option<PlaneDamageClips>,
    pub plane_claim: PlaneClaim,
    pub sync: Option<(SyncPoint, Option<Arc<OwnedFd>>)>,
}

impl<B: Buffer, F: Framebuffer> PlaneConfig<B, F> {
    #[inline]
    pub fn is_compatible(&self, other: &PlaneConfig<B, F>) -> bool {
        self.properties.is_compatible(&other.properties)
    }
}

impl<B: Buffer, F: Framebuffer> Clone for PlaneConfig<B, F> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            properties: self.properties,
            buffer: self.buffer.clone(),
            damage_clips: self.damage_clips.clone(),
            plane_claim: self.plane_claim.clone(),
            sync: self.sync.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct PlaneElementState {
    id: Id,
    commit: CommitCounter,
    z_index: usize,
    cursor_size: Option<Size<i32, Physical>>,
}

#[derive(Debug)]
struct PlaneState<B: Buffer, F: Framebuffer> {
    skip: bool,
    needs_test: bool,
    element_state: Option<PlaneElementState>,
    config: Option<PlaneConfig<B, F>>,
}

impl<B: Buffer, F: Framebuffer> Default for PlaneState<B, F> {
    #[inline]
    fn default() -> Self {
        Self {
            skip: true,
            needs_test: false,
            element_state: Default::default(),
            config: Default::default(),
        }
    }
}

impl<B: Buffer, F: Framebuffer> PlaneState<B, F> {
    #[inline]
    fn buffer(&self) -> Option<&DrmScanoutBuffer<B, F>> {
        self.config.as_ref().map(|config| &config.buffer)
    }

    #[inline]
    fn is_compatible(&self, other: &Self) -> bool {
        match (self.config.as_ref(), other.config.as_ref()) {
            (Some(a), Some(b)) => a.is_compatible(b),
            (None, None) => true,
            _ => false,
        }
    }

    fn async_flip_plane(&self) -> AsyncFlipPlane {
        let config = self.config.as_ref();
        AsyncFlipPlane {
            config: config.map(|config| (*config.buffer.as_ref(), config.properties)),
            skip: self.skip,
            client_buffer: config
                .is_some_and(|config| matches!(config.buffer.buffer, ScanoutBuffer::Wayland(_))),
        }
    }
}

impl<B: Buffer, F: Framebuffer> Clone for PlaneState<B, F> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            skip: self.skip,
            needs_test: self.needs_test,
            element_state: self.element_state.clone(),
            config: self.config.clone(),
        }
    }
}

#[derive(Debug)]
struct FrameState<B: Buffer, F: Framebuffer> {
    planes: SmallVec<[(plane::Handle, PlaneState<B, F>); 10]>,
}

impl<B: Buffer, F: Framebuffer> FrameState<B, F> {
    #[inline]
    fn is_assigned(&self, handle: plane::Handle) -> bool {
        self.planes
            .iter()
            .find_map(|(p, state)| {
                if *p == handle {
                    Some(state.config.is_some())
                } else {
                    None
                }
            })
            .unwrap_or(false)
    }

    #[inline]
    fn overlaps(&self, handle: plane::Handle, element_geometry: Rectangle<i32, Physical>) -> bool {
        self.planes
            .iter()
            .find(|(p, _)| *p == handle)
            .and_then(|(_, state)| {
                state
                    .config
                    .as_ref()
                    .map(|config| config.properties.dst.overlaps(element_geometry))
            })
            .unwrap_or(false)
    }

    #[inline]
    fn plane_state(&self, handle: plane::Handle) -> Option<&PlaneState<B, F>> {
        self.planes
            .iter()
            .find_map(|(p, state)| if *p == handle { Some(state) } else { None })
    }

    #[inline]
    fn plane_state_mut(&mut self, handle: plane::Handle) -> Option<&mut PlaneState<B, F>> {
        self.planes
            .iter_mut()
            .find_map(|(p, state)| if *p == handle { Some(state) } else { None })
    }

    #[inline]
    fn plane_properties(&self, handle: plane::Handle) -> Option<&PlaneProperties> {
        self.plane_state(handle)
            .and_then(|state| state.config.as_ref())
            .map(|config| &config.properties)
    }

    #[inline]
    fn plane_buffer(&self, handle: plane::Handle) -> Option<&DrmScanoutBuffer<B, F>> {
        self.plane_state(handle)
            .and_then(|state| state.config.as_ref().map(|config| &config.buffer))
    }
}

impl<B: Buffer, F: Framebuffer> FrameState<B, F> {
    fn from_planes(primary_plane: plane::Handle, planes: &Planes) -> Self {
        let mut tmp = SmallVec::with_capacity(planes.overlay.len() + planes.cursor.len() + 1);
        tmp.push((primary_plane, PlaneState::default()));
        tmp.extend(
            planes
                .cursor
                .iter()
                .map(|info| (info.handle, PlaneState::default())),
        );
        tmp.extend(
            planes
                .overlay
                .iter()
                .map(|info| (info.handle, PlaneState::default())),
        );

        FrameState { planes: tmp }
    }
}

impl<B: Buffer, F: Framebuffer> FrameState<B, F> {
    #[profiling::function]
    #[inline]
    fn set_state(&mut self, plane: plane::Handle, state: PlaneState<B, F>) {
        let current_config = match self.plane_state_mut(plane) {
            Some(config) => config,
            None => return,
        };
        *current_config = state;
    }

    #[profiling::function]
    fn test_state(
        &mut self,
        surface: &DrmSurface,
        supports_fencing: bool,
        plane: plane::Handle,
        state: PlaneState<B, F>,
        allow_modeset: bool,
    ) -> Result<(), DrmError> {
        let current_config = match self.plane_state_mut(plane) {
            Some(config) => config,
            None => return Ok(()),
        };
        let backup = current_config.clone();
        *current_config = state;

        let res = surface.test_state(self.build_planes(surface, supports_fencing, true), allow_modeset);

        if res.is_err() {
            // test failed, restore previous state
            *self.plane_state_mut(plane).unwrap() = backup;
        } else {
            self.planes
                .iter_mut()
                .for_each(|(_, state)| state.needs_test = false);
        }

        res
    }

    #[profiling::function]
    fn test_state_complete(
        &mut self,
        previous_frame: &Self,
        surface: &DrmSurface,
        supports_fencing: bool,
        allow_modeset: bool,
        allow_partial_update: bool,
    ) -> Result<(), DrmError> {
        let needs_test = self.planes.iter().any(|(_, state)| state.needs_test);
        let is_fully_compatible = self.planes.iter().all(|(handle, state)| {
            previous_frame
                .plane_state(*handle)
                .map(|other| state.is_compatible(other))
                .unwrap_or(false)
        });

        if allow_partial_update && (!needs_test || is_fully_compatible) {
            trace!("skipping fully compatible state test");
            self.planes
                .iter_mut()
                .for_each(|(_, state)| state.needs_test = false);
            return Ok(());
        }

        let res = surface.test_state(
            self.build_planes(surface, supports_fencing, allow_partial_update),
            allow_modeset,
        );

        if res.is_ok() {
            self.planes
                .iter_mut()
                .for_each(|(_, state)| state.needs_test = false);
        }

        res
    }

    #[profiling::function]
    fn commit(
        &mut self,
        surface: &DrmSurface,
        supports_fencing: bool,
        allow_partial_update: bool,
        event: bool,
    ) -> Result<(), crate::backend::drm::error::Error> {
        debug_assert!(!self.planes.iter().any(|(_, state)| state.needs_test));
        surface.commit(
            self.build_planes(surface, supports_fencing, allow_partial_update),
            event,
        )
    }

    #[profiling::function]
    fn page_flip(
        &mut self,
        surface: &DrmSurface,
        supports_fencing: bool,
        allow_partial_update: bool,
        event: bool,
    ) -> Result<(), crate::backend::drm::error::Error> {
        debug_assert!(!self.planes.iter().any(|(_, state)| state.needs_test));
        surface.page_flip(
            self.build_planes(surface, supports_fencing, allow_partial_update),
            event,
        )
    }

    /// The planes of this frame as [`async_flip_blocker`] takes them, `current` is the frame on screen
    fn async_flip_planes<'a>(
        &'a self,
        current: &'a Self,
    ) -> impl Iterator<Item = (plane::Handle, Option<AsyncFlipPlane>, AsyncFlipPlane)> + 'a {
        self.planes.iter().map(|(handle, next)| {
            (
                *handle,
                current.plane_state(*handle).map(PlaneState::async_flip_plane),
                next.async_flip_plane(),
            )
        })
    }

    /// Flips the primary plane to its framebuffer of this frame without waiting for the vblank,
    /// leaving everything else as it is (see [`DrmSurface::page_flip_async`])
    #[profiling::function]
    fn page_flip_async(
        &self,
        surface: &DrmSurface,
        primary: plane::Handle,
    ) -> Result<(), crate::backend::drm::error::Error> {
        debug_assert!(!self.planes.iter().any(|(_, state)| state.needs_test));
        let config = self
            .plane_state(primary)
            .and_then(|state| state.config.as_ref())
            .ok_or(crate::backend::drm::error::Error::NoFramebuffer(primary))?;
        if matches!(config.buffer.buffer, ScanoutBuffer::Swapchain(_)) {
            // A composited frame (`ALLOW_ASYNC_PAGE_FLIP_COMPOSITED`): the commit carries no fence,
            // so the rendering has to be done before the kernel may scan the buffer out
            if let Some((sync, _)) = config.sync.as_ref() {
                sync.wait().map_err(|_| {
                    crate::backend::drm::error::Error::Access(crate::backend::drm::error::AccessError {
                        errmsg: "Waiting for the rendering of an async page flip was interrupted",
                        dev: surface.dev_path(),
                        source: std::io::Error::from(std::io::ErrorKind::Interrupted),
                    })
                })?;
            }
            return surface.page_flip_async(*config.buffer.as_ref(), None);
        }
        // The fence a vsynced flip of this frame would set as `IN_FENCE_FD` (see `build_planes`)
        let fence = config
            .sync
            .as_ref()
            .and_then(|(_, fence)| fence.as_ref().map(|fence| fence.as_fd()));
        surface.page_flip_async(*config.buffer.as_ref(), fence)
    }

    #[profiling::function]
    fn build_planes<'a>(
        &'a mut self,
        surface: &'a DrmSurface,
        supports_fencing: bool,
        allow_partial_update: bool,
    ) -> impl IntoIterator<Item = super::PlaneState<'a>> {
        for (_, state) in self.planes.iter_mut().filter(|(_, state)| !state.skip) {
            if let Some(config) = state.config.as_mut() {
                // Try to extract a native fence out of the supplied sync point if any
                // If the sync point has no native fence or the surface does not support
                // fencing force a wait
                if let Some((sync, fence)) = config.sync.as_mut() {
                    if supports_fencing && fence.is_none() {
                        *fence = sync.export().map(Arc::new);
                    }
                }
            }
        }

        self.planes
            .iter_mut()
            .filter(move |(handle, state)| {
                // If we are not allowed to do an partial update we want to update all
                // planes we can claim. This makes sure we also reset planes we never
                // actually used. We can skip getting a claim here if we have a
                // config as this means we already claimed the plane for us.
                if allow_partial_update {
                    // A partial update would technically only have to include planes that
                    // actually changed. This includes planes we previously used and have to
                    // reset and planes we use and want to update.
                    // Both is already encoded into state.skip, so this should be the only
                    // thing we have to consider here.
                    //
                    // But...Unfortunately some drivers seem to have issues with partial
                    // updates, at least when it does not contain the primary plane, resulting
                    // in strange issues like e.g. repeating plane content, side-scrolling planes,
                    // wrapping planes around edges...
                    //
                    // So until these things are fixed just always send the whole state. We do not
                    // have to send planes we never used, but we include planes we want to reset or
                    // that explicitly changed represented by !state.skip and all planes currently in
                    // use represented by having an config defined.
                    !state.skip || state.config.is_some()
                } else {
                    state.config.is_some() || surface.claim_plane(*handle).is_some()
                }
            })
            .map(move |(handle, state)| super::surface::PlaneState {
                handle: *handle,
                config: state.config.as_mut().map(|config| super::PlaneConfig {
                    src: config.properties.src,
                    dst: config.properties.dst,
                    alpha: config.properties.alpha,
                    transform: config.properties.transform,
                    damage_clips: config.damage_clips.as_ref().map(|d| d.blob()),
                    fb: *config.buffer.as_ref(),
                    fence: config
                        .sync
                        .as_ref()
                        .and_then(|(_, fence)| fence.as_ref().map(|fence| fence.as_fd())),
                }),
            })
    }
}

type CompositorFrameState<A, F> =
    FrameState<<A as Allocator>::Buffer, <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer>;

type FrameErrorType<A, F> = FrameError<
    <A as Allocator>::Error,
    <<A as Allocator>::Buffer as AsDmabuf>::Error,
    <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Error,
>;

pub(crate) type FrameResult<T, A, F> = Result<T, FrameErrorType<A, F>>;

pub(crate) type RenderFrameErrorType<A, F, R> = RenderFrameError<
    <A as Allocator>::Error,
    <<A as Allocator>::Buffer as AsDmabuf>::Error,
    <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Error,
    <R as RendererSuper>::Error,
>;

#[derive(Debug)]
struct CursorState<G: AsFd + 'static> {
    allocator: GbmAllocator<G>,
    framebuffer_exporter: GbmFramebufferExporter<G>,
    previous_output_transform: Option<Transform>,
    previous_output_scale: Option<Scale<f64>>,
    #[cfg(feature = "renderer_pixman")]
    pixman_renderer: Option<PixmanRenderer>,
}

#[derive(Debug, thiserror::Error, Copy, Clone)]
enum ExportBufferError {
    #[error("the buffer has no underlying storage")]
    NoUnderlyingStorage,
    #[error("exporting the framebuffer failed")]
    ExportFailed,
    #[error("no framebuffer could be exported")]
    Unsupported,
}

impl From<ExportBufferError> for Option<RenderingReason> {
    #[inline]
    fn from(err: ExportBufferError) -> Self {
        if matches!(err, ExportBufferError::ExportFailed) {
            // Export failed could mean the buffer could
            // not be used to add a drm framebuffer. This
            // especially can happen on kmsro devices where
            // a buffer format not usable for scan-out can
            // not be used to add a framebuffer
            // We can try to give the client another chance
            // by announcing a scan-out tranche
            Some(RenderingReason::ScanoutFailed)
        } else {
            // We provide no reason for rendering here as there
            // is no action that can be taken to make it work
            None
        }
    }
}

#[derive(Debug)]
struct OverlayPlaneElementIds {
    plane_ids: Vec<(plane::Handle, Id, Id)>,
}

impl OverlayPlaneElementIds {
    fn from_planes(planes: &Planes) -> Self {
        let overlay_plane_count = planes.overlay.len();

        Self {
            plane_ids: Vec::with_capacity(overlay_plane_count),
        }
    }

    fn plane_id_for_element_id(&mut self, plane: &plane::Handle, element_id: &Id) -> Id {
        // Either get the existing plane id for the plane when the stored element id
        // matches or generate a new Id (and update the element id)
        let existing = self.plane_ids.iter_mut().find(|(p, _, _)| p == plane);
        if let Some((_, plane_id, current_element_id)) = existing {
            if current_element_id != element_id {
                *plane_id = Id::new();
                *current_element_id = element_id.clone();
            }

            plane_id.clone()
        } else {
            let plane_id = Id::new();

            self.plane_ids
                .push((*plane, plane_id.clone(), element_id.clone()));

            plane_id
        }
    }

    fn contains_plane_id(&self, plane_id: &Id) -> bool {
        self.plane_ids.iter().any(|(_, p, _)| p == plane_id)
    }

    fn remove_plane(&mut self, plane: &plane::Handle) {
        self.plane_ids.retain(|(p, _, _)| p != plane);
    }
}

struct PlaneAssignment {
    handle: plane::Handle,
    type_: PlaneType,
}

impl From<&PlaneInfo> for PlaneAssignment {
    #[inline]
    fn from(value: &PlaneInfo) -> Self {
        PlaneAssignment {
            handle: value.handle,
            type_: value.type_,
        }
    }
}

struct PendingFrame<A: Allocator, F: ExportFramebuffer<<A as Allocator>::Buffer>, U> {
    frame: CompositorFrameState<A, F>,
    user_data: U,
    presentation_mode: PresentationMode,
    async_refused: bool,
}

impl<A, F, U> std::fmt::Debug for PendingFrame<A, F, U>
where
    A: Allocator,
    <A as Allocator>::Buffer: std::fmt::Debug,
    F: ExportFramebuffer<<A as Allocator>::Buffer>,
    <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer: std::fmt::Debug,
    U: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingFrame")
            .field("frame", &self.frame)
            .field("user_data", &self.user_data)
            .field("presentation_mode", &self.presentation_mode)
            .field("async_refused", &self.async_refused)
            .finish()
    }
}

struct QueuedFrame<A: Allocator, F: ExportFramebuffer<<A as Allocator>::Buffer>, U> {
    prepared_frame: PreparedFrame<A, F>,
    user_data: U,
}

impl<A, F, U> std::fmt::Debug for QueuedFrame<A, F, U>
where
    A: Allocator,
    <A as Allocator>::Buffer: std::fmt::Debug,
    F: ExportFramebuffer<<A as Allocator>::Buffer>,
    <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer: std::fmt::Debug,
    U: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueuedFrame")
            .field("prepared_frame", &self.prepared_frame)
            .field("user_data", &self.user_data)
            .finish()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum PreparedFrameKind {
    Full,
    Partial,
}

struct PreparedFrame<A: Allocator, F: ExportFramebuffer<<A as Allocator>::Buffer>> {
    frame: CompositorFrameState<A, F>,
    kind: PreparedFrameKind,
    /// Rendered with [`FrameFlags::ALLOW_ASYNC_PAGE_FLIP`]
    async_page_flip: bool,
    /// Rendered with [`FrameFlags::ALLOW_ASYNC_PAGE_FLIP_COMPOSITED`] too
    async_composited: bool,
}

impl<A: Allocator, F: ExportFramebuffer<<A as Allocator>::Buffer>> PreparedFrame<A, F> {
    #[inline]
    fn is_empty(&self) -> bool {
        // It can happen that we have no changes, but there is a pending commit or
        // we are forced to do a full update in which case we just set the previous state again
        self.kind == PreparedFrameKind::Partial && self.frame.planes.iter().all(|p| p.1.skip)
    }
}

impl<A, F> std::fmt::Debug for PreparedFrame<A, F>
where
    A: Allocator,
    <A as Allocator>::Buffer: std::fmt::Debug,
    F: ExportFramebuffer<<A as Allocator>::Buffer>,
    <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedFrame")
            .field("frame", &self.frame)
            .field("kind", &self.kind)
            .field("async_page_flip", &self.async_page_flip)
            .field("async_composited", &self.async_composited)
            .finish()
    }
}

bitflags::bitflags! {
    /// Possible flags for a DMA buffer
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct FrameFlags: u32 {
        /// Allow to realize the frame by scanning out elements on the primary plane
        /// with the same pixel format as the main swapchain
        const ALLOW_PRIMARY_PLANE_SCANOUT = 1;
        /// Allow to realize the frame by scanning out elements on the primary plane
        /// regardless of their format
        const ALLOW_PRIMARY_PLANE_SCANOUT_ANY = 2;
        /// Allow to realize the frame by scanning out elements on overlay planes
        const ALLOW_OVERLAY_PLANE_SCANOUT = 4;
        /// Allow to realize the frame by scanning out elements on cursor planes
        const ALLOW_CURSOR_PLANE_SCANOUT = 8;
        /// Return `EmptyFrame`, if only the cursor plane would have been updated
        const SKIP_CURSOR_ONLY_UPDATES = 16;
        /// Present the frame with an async page flip (`DRM_MODE_PAGE_FLIP_ASYNC`, tearing) if it qualifies
        ///
        /// No overlay plane is used in a frame rendered with this flag. When the frame is submitted,
        /// it flips without waiting for the vertical blank if all of these hold, and at the next
        /// vertical blank like any other frame otherwise:
        /// - the device supports it ([`DrmDevice::supports_async_page_flip`](crate::backend::drm::DrmDevice::supports_async_page_flip));
        /// - no state change is pending ([`DrmSurface::commit_pending`]), the frame is not the
        ///   first one after a reset, and no VRR switch made with [`DrmSurface::use_vrr`] still has to
        ///   be committed (an async commit cannot change `VRR_ENABLED`, so this frame sends it);
        /// - the primary plane scans out a client buffer directly (no composition);
        /// - its source and destination rectangle, transform, alpha and format (with the modifier)
        ///   are those of the primary plane on screen;
        /// - no other plane changes: the cursor plane may be in use, but neither moves nor changes.
        ///
        /// Like vsynced direct scan-out, this relies on the acquire point of an explicitly
        /// synchronized client buffer having signaled before the frame is rendered (as with
        /// `DrmSyncobjBlocker`): the async commit carries the same pre-signaled `IN_FENCE_FD` as a
        /// vsynced one, where the kernel takes it (see [`DrmSurface::page_flip_async`]).
        ///
        /// The async commit sets only the primary plane's framebuffer (and fence), see
        /// [`DrmSurface::page_flip_async`] for the kernel rules. If the kernel refuses it with
        /// `EINVAL`, the same frame is committed again at once to flip at the next vertical blank,
        /// so no frame is lost. Any other error, like `EBUSY` while a previous commit is still in
        /// flight, is returned like the error of any other page flip.
        /// [`DrmCompositor::pending_frame`] tells how a submitted frame went out.
        ///
        /// The flag applies to the frame rendered with it only. Without it nothing changes, and
        /// the legacy api ignores it.
        const ALLOW_ASYNC_PAGE_FLIP = 32;
        /// With [`FrameFlags::ALLOW_ASYNC_PAGE_FLIP`]: a frame whose primary plane shows the
        /// composited swapchain buffer may flip async as well, not only one that scans out a client
        /// buffer directly. Every other rule of [`FrameFlags::ALLOW_ASYNC_PAGE_FLIP`] still applies:
        /// the swapchain buffer has to fit the primary plane on screen (it does from the second
        /// composited frame on, as long as the mode and the swapchain's format stay), and no other
        /// plane may change, so a cursor that moves on the cursor plane makes the frame flip at the
        /// next vertical blank (leave out [`FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT`] to draw the
        /// cursor into the frame instead).
        ///
        /// The async commit of a swapchain buffer carries no `IN_FENCE_FD` (Linux 6.8 to 6.10 refuse
        /// one, and implicit fencing of the buffer is not something every driver provides), so the
        /// submit waits on the CPU until the rendering of the frame has finished before it commits:
        /// [`DrmCompositor::queue_frame`] blocks for that long. The buffer is complete when the
        /// kernel scans it out either way. Without [`FrameFlags::ALLOW_ASYNC_PAGE_FLIP`] this flag
        /// does nothing.
        const ALLOW_ASYNC_PAGE_FLIP_COMPOSITED = 64;
        /// Allow to realize the frame by assigning elements on any plane
        const ALLOW_SCANOUT = Self::ALLOW_PRIMARY_PLANE_SCANOUT.bits() | Self::ALLOW_OVERLAY_PLANE_SCANOUT.bits() | Self::ALLOW_CURSOR_PLANE_SCANOUT.bits();
        /// Safe default set of flags
        const DEFAULT = Self::ALLOW_SCANOUT.bits();
    }
}

/// A frame that was committed and waits for its page flip, see [`DrmCompositor::pending_frame`]
#[derive(Debug)]
pub struct FrameRef<'a, U> {
    /// User data of the frame, as given to [`DrmCompositor::queue_frame`]
    pub user_data: &'a U,
    /// How the frame was committed: [`PresentationMode::Async`] if with an async page flip
    /// ([`FrameFlags::ALLOW_ASYNC_PAGE_FLIP`]), [`PresentationMode::VSync`] otherwise.
    ///
    /// A driver may still carry out an async commit at the vertical blank: amdgpu does for a
    /// buffer in another memory domain than the one on screen (up to Linux 7.1), and i915
    /// (Skylake and newer) for the first async flip after a vsynced one.
    pub presentation_mode: PresentationMode,
    /// The frame qualified for an async page flip, but the kernel refused the async commit with
    /// `EINVAL`, so the same frame was committed again to flip at the next vertical blank
    pub async_refused: bool,
}

/// Why a frame with [`FrameFlags::ALLOW_ASYNC_PAGE_FLIP`] flips at the next vertical blank
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AsyncFlipBlocker {
    /// The device does not support async page flips, or uses the legacy api
    Unsupported,
    /// The frame is the first one after a reset or comes with a state change
    FullCommit,
    /// A VRR switch (or a reset of the surface) still has to be committed, which an async commit
    /// cannot do
    VrrChange,
    /// The primary plane is off, or composited without
    /// [`FrameFlags::ALLOW_ASYNC_PAGE_FLIP_COMPOSITED`]
    NoDirectScanout,
    /// The primary plane changes more than its framebuffer
    PrimaryChanged,
    /// Another plane changes
    OtherPlaneChanged,
}

/// The surface a frame is submitted to, as far as async page flips are concerned
#[derive(Debug, Clone, Copy)]
struct AsyncFlipSurface {
    /// The device does async page flips (never on the legacy api)
    supported: bool,
    /// The frame is the first one after a reset or comes with a state change
    full_commit: bool,
    /// `VRR_ENABLED` still has to be committed, see `DrmSurface::vrr_commit_pending`
    vrr_commit_pending: bool,
    /// The frame may flip async with the composited swapchain buffer on the primary plane
    /// ([`FrameFlags::ALLOW_ASYNC_PAGE_FLIP_COMPOSITED`])
    composited: bool,
}

/// One plane of a frame, as far as async page flips are concerned
#[derive(Debug, Clone, Copy, PartialEq)]
struct AsyncFlipPlane {
    /// Framebuffer and properties, `None` if the plane is off
    config: Option<(framebuffer::Handle, PlaneProperties)>,
    /// A partial update leaves the plane out
    skip: bool,
    /// The plane scans out a client buffer
    client_buffer: bool,
}

/// Checks whether a frame can be committed as an async page flip on top of the frame on screen:
/// the surface allows it, the `primary` plane scans out a client buffer (or the swapchain, if
/// `surface.composited`) and changes nothing but the framebuffer, and no other plane changes at
/// all. Each plane is given as `(handle, on screen, next)`.
fn async_flip_blocker(
    surface: AsyncFlipSurface,
    primary: plane::Handle,
    planes: impl IntoIterator<Item = (plane::Handle, Option<AsyncFlipPlane>, AsyncFlipPlane)>,
) -> Option<AsyncFlipBlocker> {
    if !surface.supported {
        return Some(AsyncFlipBlocker::Unsupported);
    }
    if surface.full_commit {
        return Some(AsyncFlipBlocker::FullCommit);
    }
    if surface.vrr_commit_pending {
        return Some(AsyncFlipBlocker::VrrChange);
    }

    // Comparing the configs, not only `skip`: `SKIP_CURSOR_ONLY_UPDATES` skips a moved cursor
    let unchanged = |current: Option<AsyncFlipPlane>, next: AsyncFlipPlane| {
        next.skip && current.and_then(|current| current.config) == next.config
    };
    let mut primary_plane = None;
    let mut other_plane_changed = false;
    for (handle, current, next) in planes {
        if handle == primary {
            primary_plane = Some((current, next));
        } else if !unchanged(current, next) {
            other_plane_changed = true;
        }
    }

    let Some((current, next)) = primary_plane else {
        return Some(AsyncFlipBlocker::NoDirectScanout);
    };
    let Some((_, properties)) = next.config.filter(|_| next.client_buffer || surface.composited) else {
        return Some(AsyncFlipBlocker::NoDirectScanout);
    };
    if !current
        .and_then(|current| current.config)
        .is_some_and(|(_, current)| current.is_compatible(&properties))
    {
        return Some(AsyncFlipBlocker::PrimaryChanged);
    }
    if other_plane_changed {
        return Some(AsyncFlipBlocker::OtherPlaneChanged);
    }
    None
}

/// How a frame was page-flipped, see [`flip_frame`]
#[derive(Debug)]
struct FlipOutcome {
    result: Result<(), DrmError>,
    presentation_mode: PresentationMode,
    async_refused: bool,
}

/// Page-flips a frame that needs no full commit, with an async page flip if `try_async`, and at
/// the next vertical blank otherwise. `flip` commits the frame in the given mode.
///
/// If the kernel refuses the async commit with `EINVAL` (the property and driver checks), the
/// frame is flipped vsynced at once, so it is not lost, and the refusal is not returned: it never
/// reaches `handle_flip`, which would rule out direct scan-out of the element on the primary plane
/// for the next frames. Any other error of the async commit is returned as it is, `EBUSY` (a
/// previous commit still in flight) included, since a vsynced non-blocking commit would get it too.
fn flip_frame(
    try_async: bool,
    mut flip: impl FnMut(PresentationMode) -> Result<(), DrmError>,
) -> FlipOutcome {
    let mut async_refused = false;
    if try_async {
        match flip(PresentationMode::Async) {
            Err(err) if err.is_invalid_argument() => {
                debug!(?err, "async page flip refused, flipping at the next vblank");
                async_refused = true;
            }
            result => {
                return FlipOutcome {
                    presentation_mode: if result.is_ok() {
                        PresentationMode::Async
                    } else {
                        PresentationMode::VSync
                    },
                    result,
                    async_refused,
                };
            }
        }
    }
    FlipOutcome {
        result: flip(PresentationMode::VSync),
        presentation_mode: PresentationMode::VSync,
        async_refused,
    }
}

/// Composite an output using a combination of planes and rendering
///
/// see the [`module docs`](crate::backend::drm::compositor) for more information
#[derive(Debug)]
pub struct DrmCompositor<A, F, U, G>
where
    A: Allocator,
    F: ExportFramebuffer<A::Buffer>,
    <F as ExportFramebuffer<A::Buffer>>::Framebuffer: std::fmt::Debug + Send + Sync + 'static,
    G: AsFd + 'static,
{
    output_mode_source: OutputModeSource,
    surface: Arc<DrmSurface>,
    planes: Planes,
    overlay_plane_element_ids: OverlayPlaneElementIds,
    damage_tracker: OutputDamageTracker,
    primary_is_opaque: bool,
    primary_plane_element_id: Id,
    primary_plane_damage_bag: DamageBag<i32, BufferCoords>,
    supports_fencing: bool,
    reset_pending: bool,
    signaled_fence: Option<Arc<OwnedFd>>,

    framebuffer_exporter: F,

    current_frame: CompositorFrameState<A, F>,
    pending_frame: Option<PendingFrame<A, F, U>>,
    queued_frame: Option<QueuedFrame<A, F, U>>,
    next_frame: Option<PreparedFrame<A, F>>,

    swapchain: Swapchain<A>,

    cursor_size: Size<i32, Physical>,
    cursor_state: Option<CursorState<G>>,

    element_states: IndexMap<Id, ElementState<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>,
    previous_element_states: IndexMap<Id, ElementState<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>,
    opaque_regions: Vec<Rectangle<i32, Physical>>,
    element_opaque_regions_workhouse: Vec<Rectangle<i32, Physical>>,

    debug_flags: DebugFlags,
    span: tracing::Span,
}

impl<A, F, U, G> DrmCompositor<A, F, U, G>
where
    A: Allocator,
    <A as Allocator>::Error: std::error::Error + Send + Sync,
    <A as Allocator>::Buffer: AsDmabuf,
    <A::Buffer as AsDmabuf>::Error: std::error::Error + Send + Sync + std::fmt::Debug,
    F: ExportFramebuffer<A::Buffer>,
    <F as ExportFramebuffer<A::Buffer>>::Framebuffer: std::fmt::Debug + Send + Sync + 'static,
    <F as ExportFramebuffer<A::Buffer>>::Error: std::error::Error + Send + Sync,
    G: AsFd + Clone,
{
    /// Initialize a new [`DrmCompositor`].
    ///
    /// The [`OutputModeSource`] can be created from an [`Output`](crate::output::Output), which will automatically track
    /// the output's mode changes. An [`OutputModeSource::Static`] variant should only be used when
    /// manually updating modes using [`DrmCompositor::set_output_mode_source`].
    ///
    /// - `output_mode_source` is used to determine the current mode, scale and transform
    /// - `surface` for the compositor to use
    /// - `planes` defines which planes the compositor is allowed to use for direct scan-out.
    ///           `None` will result in the compositor to use all planes as specified by [`DrmSurface::planes`]
    /// - `allocator` used for the primary plane swapchain
    /// - `color_formats` are tested in order until a working configuration is found
    /// - `renderer_formats` as reported by the used renderer, used to build the intersection between
    ///                      the possible scan-out formats of the primary plane and the renderer
    /// - `framebuffer_exporter` is used to create drm framebuffers for the swapchain buffers (and if possible
    ///                          for element buffers) for scan-out
    /// - `cursor_size` as reported by the drm device, used for creating buffer for the cursor plane
    /// - `gbm` device used for creating buffers for the cursor plane, `None` will disable the cursor plane
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip_all)]
    pub fn new(
        output_mode_source: impl Into<OutputModeSource> + Debug,
        surface: DrmSurface,
        planes: Option<Planes>,
        mut allocator: A,
        framebuffer_exporter: F,
        color_formats: impl IntoIterator<Item = DrmFourcc>,
        renderer_formats: impl IntoIterator<Item = DrmFormat>,
        cursor_size: Size<u32, BufferCoords>,
        gbm: Option<GbmDevice<G>>,
    ) -> FrameResult<Self, A, F> {
        let signaled_fence = match surface.create_syncobj(true) {
            Ok(signaled_syncobj) => match surface.syncobj_to_fd(signaled_syncobj, true) {
                Ok(signaled_fence) => {
                    let _ = surface.destroy_syncobj(signaled_syncobj);
                    Some(Arc::new(signaled_fence))
                }
                Err(err) => {
                    tracing::warn!(?err, "failed to export signaled syncobj");
                    let _ = surface.destroy_syncobj(signaled_syncobj);
                    None
                }
            },
            Err(err) => {
                tracing::warn!(?err, "failed to create signaled syncobj");
                None
            }
        };

        let span = info_span!(
            parent: None,
            "drm_compositor",
            device = ?surface.dev_path(),
            crtc = ?surface.crtc(),
        );

        let output_mode_source = output_mode_source.into();
        let renderer_formats = renderer_formats.into_iter().collect::<Vec<_>>();

        let mut error = None;
        let surface = Arc::new(surface);
        let mut planes = match planes {
            Some(planes) => planes,
            None => surface.planes().clone(),
        };

        // We do not support direct scan-out on legacy
        if surface.is_legacy() {
            planes.cursor.clear();
            planes.overlay.clear();
        }

        // The selection algorithm expects the planes to be ordered form front to back
        planes
            .overlay
            .sort_by_key(|p| std::cmp::Reverse(p.zpos.unwrap_or_default()));

        let driver = surface.get_driver().map_err(|err| {
            FrameError::DrmError(DrmError::Access(AccessError {
                errmsg: "Failed to query drm driver",
                dev: surface.dev_path(),
                source: err,
            }))
        })?;
        // `IN_FENCE_FD` makes commit fail on Nvidia driver
        // https://github.com/NVIDIA/open-gpu-kernel-modules/issues/622
        let is_nvidia = driver.name().to_string_lossy().to_lowercase().contains("nvidia")
            || driver
                .description()
                .to_string_lossy()
                .to_lowercase()
                .contains("nvidia");

        let cursor_size = Size::from((cursor_size.w as i32, cursor_size.h as i32));
        let damage_tracker = OutputDamageTracker::from_mode_source(output_mode_source.clone());
        let supports_fencing = !surface.is_legacy()
            && surface
                .get_driver_capability(DriverCapability::SyncObj)
                .map(|val| val != 0)
                .map_err(|err| {
                    FrameError::DrmError(DrmError::Access(AccessError {
                        errmsg: "Failed to query driver capability",
                        dev: surface.dev_path(),
                        source: err,
                    }))
                })?
            && plane_has_property(&*surface, surface.plane(), "IN_FENCE_FD")?
            && !(is_nvidia && nvidia_drm_version().unwrap_or((0, 0, 0)) < (560, 35, 3));

        for format in color_formats {
            debug!("Testing color format: {}", format);
            match Self::find_supported_format(
                surface.clone(),
                supports_fencing,
                &planes,
                allocator,
                &framebuffer_exporter,
                renderer_formats.clone(),
                format,
            ) {
                Ok((swapchain, is_opaque)) => {
                    let cursor_state = gbm.map(|gbm| {
                        #[cfg(feature = "renderer_pixman")]
                        let pixman_renderer = match PixmanRenderer::new() {
                            Ok(pixman_renderer) => Some(pixman_renderer),
                            Err(err) => {
                                tracing::warn!(?err, "failed to initialize pixman renderer for cursor plane");
                                None
                            }
                        };

                        let cursor_allocator =
                            GbmAllocator::new(gbm.clone(), GbmBufferFlags::CURSOR | GbmBufferFlags::WRITE);
                        let framebuffer_exporter = GbmFramebufferExporter::new(gbm.clone(), NodeFilter::None);
                        CursorState {
                            allocator: cursor_allocator,
                            framebuffer_exporter,
                            previous_output_scale: None,
                            previous_output_transform: None,
                            #[cfg(feature = "renderer_pixman")]
                            pixman_renderer,
                        }
                    });

                    let overlay_plane_element_ids = OverlayPlaneElementIds::from_planes(&planes);
                    let current_frame = FrameState::from_planes(surface.plane(), &planes);

                    let drm_renderer = DrmCompositor {
                        primary_plane_element_id: Id::new(),
                        primary_plane_damage_bag: DamageBag::new(4),
                        primary_is_opaque: is_opaque,
                        reset_pending: true,
                        signaled_fence,
                        current_frame,
                        pending_frame: None,
                        queued_frame: None,
                        next_frame: None,
                        swapchain,
                        framebuffer_exporter,
                        cursor_size,
                        cursor_state,
                        surface,
                        damage_tracker,
                        output_mode_source,
                        planes,
                        overlay_plane_element_ids,
                        element_states: IndexMap::new(),
                        previous_element_states: IndexMap::new(),
                        opaque_regions: Vec::new(),
                        element_opaque_regions_workhouse: Vec::new(),
                        supports_fencing,
                        debug_flags: DebugFlags::empty(),
                        span,
                    };

                    return Ok(drm_renderer);
                }
                Err((alloc, err)) => {
                    warn!("Preferred format {} not available: {:?}", format, err);
                    allocator = alloc;
                    error = Some(err);
                }
            }
        }
        Err(error.unwrap())
    }

    /// Initialize a new [`DrmCompositor`] with a pre-selected format.
    ///
    /// The [`OutputModeSource`] can be created from an [`Output`](crate::output::Output), which will automatically track
    /// the output's mode changes. An [`OutputModeSource::Static`] variant should only be used when
    /// manually updating modes using [`DrmCompositor::set_output_mode_source`].
    ///
    /// - `output_mode_source` is used to determine the current mode, scale and transform
    /// - `surface` for the compositor to use
    /// - `planes` defines which planes the compositor is allowed to use for direct scan-out.
    ///   `None` will result in the compositor to use all planes as specified by [`DrmSurface::planes`]
    /// - `allocator` used for the primary plane swapchain
    /// - `framebuffer_exporter` is used to create drm framebuffers for the swapchain buffers (and if possible
    ///   for element buffers) for scan-out
    /// - `code` is the fixed format to initialize the framebuffer with
    /// - `modifiers` is the set of modifiers allowed, when allocating buffers with the specified color format
    /// - `cursor_size` as reported by the drm device, used for creating buffer for the cursor plane
    /// - `gbm` device used for creating buffers for the cursor plane, `None` will disable the cursor plane
    #[allow(clippy::too_many_arguments)]
    pub fn with_format(
        output_mode_source: impl Into<OutputModeSource> + Debug,
        surface: DrmSurface,
        planes: Option<Planes>,
        allocator: A,
        framebuffer_exporter: F,
        code: DrmFourcc,
        modifiers: impl IntoIterator<Item = DrmModifier>,
        cursor_size: Size<u32, BufferCoords>,
        gbm: Option<GbmDevice<G>>,
    ) -> FrameResult<Self, A, F> {
        let signaled_fence = match surface.create_syncobj(true) {
            Ok(signaled_syncobj) => match surface.syncobj_to_fd(signaled_syncobj, true) {
                Ok(signaled_fence) => {
                    let _ = surface.destroy_syncobj(signaled_syncobj);
                    Some(Arc::new(signaled_fence))
                }
                Err(err) => {
                    tracing::warn!(?err, "failed to export signaled syncobj");
                    let _ = surface.destroy_syncobj(signaled_syncobj);
                    None
                }
            },
            Err(err) => {
                tracing::warn!(?err, "failed to create signaled syncobj");
                None
            }
        };

        let span = info_span!(
            parent: None,
            "drm_compositor",
            device = ?surface.dev_path(),
            crtc = ?surface.crtc(),
        );

        let output_mode_source = output_mode_source.into();

        let surface = Arc::new(surface);
        let mut planes = match planes {
            Some(planes) => planes,
            None => surface.planes().clone(),
        };

        // We do not support direct scan-out on legacy
        if surface.is_legacy() {
            planes.cursor.clear();
            planes.overlay.clear();
        }

        // The selection algorithm expects the planes to be ordered form front to back
        planes
            .overlay
            .sort_by_key(|p| std::cmp::Reverse(p.zpos.unwrap_or_default()));

        let driver = surface.get_driver().map_err(|err| {
            FrameError::DrmError(DrmError::Access(AccessError {
                errmsg: "Failed to query drm driver",
                dev: surface.dev_path(),
                source: err,
            }))
        })?;
        // `IN_FENCE_FD` makes commit fail on Nvidia driver
        // https://github.com/NVIDIA/open-gpu-kernel-modules/issues/622
        let is_nvidia = driver.name().to_string_lossy().to_lowercase().contains("nvidia")
            || driver
                .description()
                .to_string_lossy()
                .to_lowercase()
                .contains("nvidia");

        let cursor_size = Size::from((cursor_size.w as i32, cursor_size.h as i32));
        let damage_tracker = OutputDamageTracker::from_mode_source(output_mode_source.clone());
        let supports_fencing = !surface.is_legacy()
            && surface
                .get_driver_capability(DriverCapability::SyncObj)
                .map(|val| val != 0)
                .map_err(|err| {
                    FrameError::DrmError(DrmError::Access(AccessError {
                        errmsg: "Failed to query driver capability",
                        dev: surface.dev_path(),
                        source: err,
                    }))
                })?
            && plane_has_property(&*surface, surface.plane(), "IN_FENCE_FD")?
            && !(is_nvidia && nvidia_drm_version().unwrap_or((0, 0, 0)) < (560, 35, 3));

        let (swapchain, is_opaque) = Self::test_format(
            &surface,
            supports_fencing,
            &planes,
            allocator,
            &framebuffer_exporter,
            code,
            modifiers,
        )
        .map_err(|(_, err)| err)?;

        let cursor_state = gbm.map(|gbm| {
            #[cfg(feature = "renderer_pixman")]
            let pixman_renderer = match PixmanRenderer::new() {
                Ok(pixman_renderer) => Some(pixman_renderer),
                Err(err) => {
                    tracing::warn!(?err, "failed to initialize pixman renderer for cursor plane");
                    None
                }
            };

            let cursor_allocator =
                GbmAllocator::new(gbm.clone(), GbmBufferFlags::CURSOR | GbmBufferFlags::WRITE);
            let framebuffer_exporter = GbmFramebufferExporter::new(gbm.clone(), NodeFilter::None);
            CursorState {
                allocator: cursor_allocator,
                framebuffer_exporter,
                previous_output_scale: None,
                previous_output_transform: None,
                #[cfg(feature = "renderer_pixman")]
                pixman_renderer,
            }
        });

        let overlay_plane_element_ids = OverlayPlaneElementIds::from_planes(&planes);
        let current_frame = FrameState::from_planes(surface.plane(), &planes);

        let drm_renderer = DrmCompositor {
            primary_plane_element_id: Id::new(),
            primary_plane_damage_bag: DamageBag::new(4),
            primary_is_opaque: is_opaque,
            reset_pending: true,
            signaled_fence,
            current_frame,
            pending_frame: None,
            queued_frame: None,
            next_frame: None,
            swapchain,
            framebuffer_exporter,
            cursor_size,
            cursor_state,
            surface,
            damage_tracker,
            output_mode_source,
            planes,
            overlay_plane_element_ids,
            element_states: IndexMap::new(),
            previous_element_states: IndexMap::new(),
            opaque_regions: Vec::new(),
            element_opaque_regions_workhouse: Vec::new(),
            supports_fencing,
            debug_flags: DebugFlags::empty(),
            span,
        };

        Ok(drm_renderer)
    }

    fn test_format(
        drm: &DrmSurface,
        supports_fencing: bool,
        planes: &Planes,
        allocator: A,
        framebuffer_exporter: &F,
        code: DrmFourcc,
        modifiers: impl IntoIterator<Item = DrmModifier>,
    ) -> Result<(Swapchain<A>, bool), (A, FrameErrorType<A, F>)> {
        let modifiers = modifiers.into_iter().collect::<IndexSet<_>>();
        let mut plane_formats = drm.plane_info().formats.iter().copied().collect::<IndexSet<_>>();

        let opaque_code = get_opaque(code).unwrap_or(code);
        if !plane_formats
            .iter()
            .any(|fmt| fmt.code == code || fmt.code == opaque_code)
        {
            return Err((allocator, FrameError::NoSupportedPlaneFormat));
        }
        plane_formats.retain(|fmt| fmt.code == code || fmt.code == opaque_code);

        if plane_formats.is_empty() {
            return Err((allocator, FrameError::NoSupportedPlaneFormat));
        }

        let plane_modifiers = plane_formats
            .iter()
            .map(|fmt| fmt.modifier)
            .collect::<IndexSet<_>>();

        let swapchain_modifiers = plane_modifiers
            .intersection(&modifiers)
            .copied()
            .collect::<Vec<_>>();

        if swapchain_modifiers.is_empty() {
            return Err((allocator, FrameError::NoSupportedPlaneFormat));
        }

        let mode = drm.pending_mode();

        let mut swapchain: Swapchain<A> = Swapchain::new(
            allocator,
            mode.size().0 as u32,
            mode.size().1 as u32,
            code,
            swapchain_modifiers,
        );

        // Test format
        let buffer = match swapchain.acquire() {
            Ok(buffer) => buffer.unwrap(),
            Err(err) => return Err((swapchain.allocator, FrameError::Allocator(err))),
        };

        let dmabuf = match buffer.export() {
            Ok(dmabuf) => dmabuf,
            Err(err) => {
                return Err((swapchain.allocator, FrameError::AsDmabufError(err)));
            }
        };

        let use_opaque = !plane_formats.iter().any(|f| f.code == code);
        let fb_buffer = match framebuffer_exporter.add_framebuffer(
            drm.device_fd(),
            ExportBuffer::Allocator(&buffer),
            use_opaque,
        ) {
            Ok(Some(fb_buffer)) => fb_buffer,
            Ok(None) => return Err((swapchain.allocator, FrameError::NoFramebuffer)),
            Err(err) => return Err((swapchain.allocator, FrameError::FramebufferExport(err))),
        };
        buffer
            .userdata()
            .insert_if_missing_threadsafe(|| CachedDrmFramebuffer::new(DrmFramebuffer::Exporter(fb_buffer)));

        let mode = drm.pending_mode();
        let handle = buffer
            .userdata()
            .get::<CachedDrmFramebuffer<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>()
            .unwrap()
            .clone();

        let mode_size = Size::from((mode.size().0 as i32, mode.size().1 as i32));

        let mut current_frame_state = FrameState::from_planes(drm.plane(), planes);
        let plane_claim = match drm.claim_plane(drm.plane()) {
            Some(claim) => claim,
            None => {
                warn!("failed to claim primary plane",);
                return Err((swapchain.allocator, FrameError::PrimaryPlaneClaimFailed));
            }
        };

        let plane_state = PlaneState {
            skip: false,
            needs_test: true,
            element_state: None,
            config: Some(PlaneConfig {
                properties: PlaneProperties {
                    src: Rectangle::from_size(dmabuf.size()).to_f64(),
                    dst: Rectangle::from_size(mode_size),
                    transform: Transform::Normal,
                    alpha: 1.0,
                    format: buffer.format(),
                },
                buffer: DrmScanoutBuffer {
                    buffer: ScanoutBuffer::Swapchain(Arc::new(buffer)),
                    fb: handle,
                },
                damage_clips: None,
                plane_claim,
                sync: None,
            }),
        };

        match current_frame_state.test_state(drm, supports_fencing, drm.plane(), plane_state, true) {
            Ok(_) => Ok((swapchain, use_opaque)),
            Err(err) => {
                warn!(
                    "Mode-setting failed with buffer format {:?}: {}",
                    dmabuf.format(),
                    err
                );
                Err((swapchain.allocator, err.into()))
            }
        }
    }

    fn find_supported_format(
        drm: Arc<DrmSurface>,
        supports_fencing: bool,
        planes: &Planes,
        allocator: A,
        framebuffer_exporter: &F,
        mut renderer_formats: Vec<DrmFormat>,
        code: DrmFourcc,
    ) -> Result<(Swapchain<A>, bool), (A, FrameErrorType<A, F>)> {
        // select a format
        let mut plane_formats = drm.plane_info().formats.iter().copied().collect::<IndexSet<_>>();

        let opaque_code = get_opaque(code).unwrap_or(code);
        if !plane_formats
            .iter()
            .any(|fmt| fmt.code == code || fmt.code == opaque_code)
        {
            return Err((allocator, FrameError::NoSupportedPlaneFormat));
        }
        plane_formats.retain(|fmt| fmt.code == code || fmt.code == opaque_code);
        renderer_formats.retain(|fmt| fmt.code == code);

        trace!("Plane formats: {:?}", plane_formats);
        trace!("Renderer formats: {:?}", renderer_formats);

        let plane_modifiers = plane_formats
            .iter()
            .map(|fmt| fmt.modifier)
            .collect::<IndexSet<_>>();
        let renderer_modifiers = renderer_formats
            .iter()
            .map(|fmt| fmt.modifier)
            .collect::<IndexSet<_>>();
        debug!(
            "Remaining intersected modifiers: {:?}",
            plane_modifiers
                .intersection(&renderer_modifiers)
                .collect::<IndexSet<_>>()
        );

        if plane_formats.is_empty() {
            return Err((allocator, FrameError::NoSupportedPlaneFormat));
        } else if renderer_formats.is_empty() {
            return Err((allocator, FrameError::NoSupportedRendererFormat));
        }

        let formats = {
            // Special case: if a format supports explicit LINEAR (but no implicit Modifiers)
            // and the other doesn't support any modifier, force Implicit.
            // This should at least result in a working pipeline possibly with a linear buffer,
            // but we cannot be sure.
            if (plane_formats.len() == 1
                && plane_formats.iter().next().unwrap().modifier == DrmModifier::Invalid
                && renderer_formats
                    .iter()
                    .all(|x| x.modifier != DrmModifier::Invalid)
                && renderer_formats.iter().any(|x| x.modifier == DrmModifier::Linear))
                || (renderer_formats.len() == 1
                    && renderer_formats.first().unwrap().modifier == DrmModifier::Invalid
                    && plane_formats.iter().all(|x| x.modifier != DrmModifier::Invalid)
                    && plane_formats.iter().any(|x| x.modifier == DrmModifier::Linear))
            {
                vec![DrmFormat {
                    code,
                    modifier: DrmModifier::Invalid,
                }]
            } else {
                plane_modifiers
                    .intersection(&renderer_modifiers)
                    .cloned()
                    .map(|modifier| DrmFormat { code, modifier })
                    .collect::<Vec<_>>()
            }
        };

        debug!("Testing Formats: {:?}", formats);

        let modifiers = formats.iter().map(|x| x.modifier).collect::<Vec<_>>();

        let (swapchain, use_opaque) = Self::test_format(
            &drm,
            supports_fencing,
            planes,
            allocator,
            framebuffer_exporter,
            code,
            modifiers,
        )?;

        Ok((swapchain, use_opaque))
    }

    /// Render the next frame
    ///
    /// - `elements` for this frame in front-to-back order
    /// - `frame_flags` specifies techniques allowed to realize the frame
    #[instrument(level = "trace", parent = &self.span, skip_all)]
    #[profiling::function]
    pub fn render_frame<'a, R, E>(
        &mut self,
        renderer: &mut R,
        elements: &'a [E],
        clear_color: impl Into<Color32F>,
        mut frame_flags: FrameFlags,
    ) -> Result<RenderFrameResult<'a, A::Buffer, F::Framebuffer, E>, RenderFrameErrorType<A, F, R>>
    where
        E: RenderElement<R>,
        R: Renderer + Bind<Dmabuf>,
        R::TextureId: Texture + 'static,
    {
        let mut clear_color = clear_color.into();

        // An async page flip can only change the primary plane
        let async_page_flip = frame_flags.contains(FrameFlags::ALLOW_ASYNC_PAGE_FLIP);
        let async_composited =
            async_page_flip && frame_flags.contains(FrameFlags::ALLOW_ASYNC_PAGE_FLIP_COMPOSITED);
        if async_page_flip {
            frame_flags.remove(FrameFlags::ALLOW_OVERLAY_PLANE_SCANOUT);
        }

        if !self.surface.is_active() {
            return Err(RenderFrameErrorType::<A, F, R>::PrepareFrame(
                FrameError::DrmError(DrmError::DeviceInactive),
            ));
        }

        // Just reset any next state, this will put
        // any already acquired slot back to the swapchain
        std::mem::drop(self.next_frame.take());

        // If a commit is pending we may still be able to just use a previous
        // state, but we want to queue a frame so we just fake the damage to
        // make sure queue_frame won't be skipped because of no damage
        let allow_partial_update = !self.reset_pending && !self.surface.commit_pending();

        let (current_size, output_scale, output_transform) = (&self.output_mode_source)
            .try_into()
            .map_err(OutputDamageTrackerError::OutputNoMode)?;

        // Output transform is specified in surface-rotation, so inversion gives us the
        // render transform for the output itself.
        let output_transform = output_transform.invert();

        // Geometry of the output derived from the output mode including the transform
        // This is used to calculate the intersection between elements and the output.
        // The renderer (and also the logic for direct scan-out) will take care of the
        // actual transform during rendering
        let output_geometry: Rectangle<_, Physical> =
            Rectangle::from_size(output_transform.transform_size(current_size));

        // We always acquire a buffer from the swapchain even
        // if we could end up doing direct scan-out on the primary plane.
        // The reason is that we can't know upfront and we need a framebuffer
        // on the primary plane to test overlay/cursor planes
        let primary_plane_buffer = self
            .swapchain
            .acquire()
            .map_err(FrameError::Allocator)?
            .ok_or(FrameError::NoFreeSlotsError)?;

        // It is safe to call export multiple times as the Slot will cache the dmabuf for us
        let dmabuf = primary_plane_buffer.export().map_err(FrameError::AsDmabufError)?;

        // Let's check if we already have a cached framebuffer for this Slot, if not try to export
        // it and use the Slot userdata to cache it
        let maybe_buffer = primary_plane_buffer
            .userdata()
            .get::<CachedDrmFramebuffer<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>();
        if maybe_buffer.is_none() {
            let fb_buffer = self
                .framebuffer_exporter
                .add_framebuffer(
                    self.surface.device_fd(),
                    ExportBuffer::Allocator(&primary_plane_buffer),
                    self.primary_is_opaque,
                )
                .map_err(FrameError::FramebufferExport)?
                .ok_or(FrameError::NoFramebuffer)?;
            primary_plane_buffer.userdata().insert_if_missing_threadsafe(|| {
                CachedDrmFramebuffer::new(DrmFramebuffer::Exporter(fb_buffer))
            });
        }

        // This unwrap is safe as we error out above if we were unable to export a framebuffer
        let fb = primary_plane_buffer
            .userdata()
            .get::<CachedDrmFramebuffer<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>()
            .unwrap()
            .clone();

        let mut opaque_regions: Vec<Rectangle<i32, Physical>> = std::mem::take(&mut self.opaque_regions);
        std::mem::swap(&mut self.previous_element_states, &mut self.element_states);
        let mut element_states = std::mem::take(&mut self.element_states);
        element_states.reserve(std::cmp::min(elements.len(), self.planes.overlay.len()));
        let mut render_element_states = RenderElementStates {
            states: HashMap::with_capacity(elements.len()),
        };

        // So first we want to create a clean state, for that we have to reset all overlay and cursor planes
        // to nothing. We only want to test if the primary plane alone can be used for scan-out.
        let mut next_frame_state: FrameState<
            <A as Allocator>::Buffer,
            <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer,
        > = {
            let previous_state = self
                .pending_frame
                .as_ref()
                .map(|pending| &pending.frame)
                .unwrap_or(&self.current_frame);

            // This will create an empty frame state, all planes are skipped by default
            let mut next_frame_state = FrameState::from_planes(self.surface.plane(), &self.planes);

            // We want to set skip to false on all planes that previously had something assigned so that
            // they get cleared when they are not longer used
            for (handle, plane_state) in next_frame_state.planes.iter_mut() {
                let reset_state = previous_state
                    .plane_state(*handle)
                    .map(|state| state.config.is_some())
                    .unwrap_or(false);

                if reset_state {
                    plane_state.skip = false;
                }
            }

            next_frame_state
        };

        // We want to make sure we can actually scan-out the primary plane, so
        // explicitly set skip to false
        let plane_claim = self.surface.claim_plane(self.surface.plane()).ok_or_else(|| {
            error!("failed to claim primary plane");
            FrameError::PrimaryPlaneClaimFailed
        })?;
        let primary_plane_state: PlaneState<
            <A as Allocator>::Buffer,
            <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer,
        > = PlaneState {
            skip: false,
            needs_test: false,
            element_state: None,
            config: Some(PlaneConfig {
                properties: PlaneProperties {
                    src: Rectangle::from_size(dmabuf.size()).to_f64(),
                    dst: Rectangle::from_size(current_size),
                    // NOTE: We do not apply the transform to the primary plane as this is handled by the dtr/renderer
                    transform: Transform::Normal,
                    alpha: 1.0,
                    format: primary_plane_buffer.format(),
                },
                buffer: DrmScanoutBuffer {
                    buffer: ScanoutBuffer::Swapchain(Arc::new(primary_plane_buffer)),
                    fb,
                },
                damage_clips: None,
                plane_claim,
                sync: None,
            }),
        };

        // unconditionally set the primary plane state
        // if this would fail the test we are screwed anyway
        next_frame_state.set_state(self.surface.plane(), primary_plane_state.clone());

        // This holds all elements that are visible on the output
        // A element is considered visible if it intersects with the output geometry
        // AND is not completely hidden behind opaque regions
        let mut output_elements: Vec<(&'a E, Rectangle<i32, Physical>, usize, bool)> =
            Vec::with_capacity(elements.len());

        let mut element_opaque_regions_workhouse = std::mem::take(&mut self.element_opaque_regions_workhouse);
        for (index, element) in elements.iter().enumerate() {
            let element_id = element.id();
            let element_geometry = element.geometry(output_scale);
            let element_loc = element_geometry.loc;

            // First test if the element overlaps with the output
            // if not we can skip it
            let element_output_geometry = match element_geometry.intersection(output_geometry) {
                Some(geo) => geo,
                None => continue,
            };

            // Then test if the element is completely hidden behind opaque regions
            element_opaque_regions_workhouse.clear();
            element_opaque_regions_workhouse.push(element_output_geometry);
            element_opaque_regions_workhouse = Rectangle::subtract_rects_many_in_place(
                element_opaque_regions_workhouse,
                opaque_regions.iter().copied(),
            );
            let element_visible_area = element_opaque_regions_workhouse
                .iter()
                .fold(0usize, |acc, item| acc + (item.size.w * item.size.h) as usize);

            if element_visible_area == 0 {
                // No need to draw a completely hidden element
                trace!("skipping completely obscured element {:?}", element.id());

                // We allow multiple instance of a single element, so do not
                // override the state if we already have one
                if !render_element_states.states.contains_key(element_id) {
                    render_element_states
                        .states
                        .insert(element_id.clone(), RenderElementState::skipped());
                }
                continue;
            }

            let element_opaque_regions = element.opaque_regions(output_scale);
            element_opaque_regions_workhouse.clear();
            element_opaque_regions_workhouse.push(element_output_geometry);
            element_opaque_regions_workhouse = Rectangle::subtract_rects_many_in_place(
                element_opaque_regions_workhouse,
                element_opaque_regions.iter().copied(),
            );
            let element_is_opaque = element_opaque_regions_workhouse.is_empty();

            opaque_regions.extend(
                element_opaque_regions
                    .into_iter()
                    .map(|mut region| {
                        region.loc += element_loc;
                        region
                    })
                    .filter_map(|geo| geo.intersection(output_geometry)),
            );

            // If the element is completely opaque and spans the whole output nothing below
            // will be visible. In this case we can short-cut the whole loop and just mark all
            // remaining elements as skipped.
            //
            // We also use this to special case for single pixel buffers that span the whole
            // output. If the last visible element is a solid color we can override the clear
            // color and remove the element completely. This will make the element directly above
            // this element the last element, enabling direct scan-out on the primary plane for it.
            if element_is_opaque && element_output_geometry.contains_rect(output_geometry) {
                let element_color = element.underlying_storage(renderer).and_then(|storage| {
                    if let UnderlyingStorage::Wayland(buffer) = storage {
                        single_pixel_buffer::get_single_pixel_buffer(buffer)
                            .ok()
                            .map(|spb| Color32F::from(spb.rgba32f()))
                    } else {
                        None
                    }
                });

                if let Some(color) = element_color {
                    clear_color = color;

                    render_element_states
                        .states
                        .entry(element_id.clone())
                        .and_modify(|state| {
                            if matches!(state.presentation_state, RenderElementPresentationState::Skipped) {
                                *state = RenderElementState::rendered(element_visible_area);
                            } else {
                                state.visible_area += element_visible_area;
                            }
                        })
                        .or_insert_with(|| RenderElementState::rendered(element_visible_area));
                } else {
                    output_elements.push((
                        element,
                        element_geometry,
                        element_visible_area,
                        element_is_opaque,
                    ));
                }

                for element in elements.iter().skip(index + 1) {
                    let element_id = element.id();
                    // We allow multiple instance of a single element, so do not
                    // override the state if we already have one
                    if !render_element_states.states.contains_key(element_id) {
                        render_element_states
                            .states
                            .insert(element_id.clone(), RenderElementState::skipped());
                    }
                }
                break;
            }

            output_elements.push((element, element_geometry, element_visible_area, element_is_opaque));
        }
        self.element_opaque_regions_workhouse = element_opaque_regions_workhouse;

        // This will hold the element that has been selected for direct scan-out on
        // the primary plane if any
        let mut primary_plane_scanout_element: Option<&'a E> = None;
        // This will hold all elements that have been assigned to the primary plane
        // for rendering
        let mut primary_plane_elements: Vec<&'a E> = Vec::with_capacity(elements.len());
        // This will hold the element per plane that has been assigned to a overlay/underlay
        // plane for direct scan-out
        let mut overlay_plane_elements: IndexMap<plane::Handle, &'a E> =
            IndexMap::with_capacity(self.planes.overlay.len());
        // This will hold the element assigned on the cursor plane if any
        let mut cursor_plane_element: Option<&'a E> = None;

        let output_elements_len = output_elements.len();
        for (index, (element, element_geometry, element_visible_area, element_is_opaque)) in
            output_elements.iter().enumerate()
        {
            let element_id = element.id();
            let element_geometry = *element_geometry;
            let remaining_elements = output_elements_len - index;
            let element_is_opaque = *element_is_opaque;

            // Check if we found our last item, we can try to do
            // direct scan-out on the primary plane
            // If we already assigned an element to
            // an underlay plane we will have a hole punch element
            // on the primary plane, this will disable direct scan-out
            // on the primary plane.
            let try_assign_primary_plane = if remaining_elements == 1 && primary_plane_elements.is_empty() {
                let crtc_background_matches_clear_color =
                    (clear_color.r() == 0f32 && clear_color.g() == 0f32 && clear_color.b() == 0f32)
                        || clear_color.a() == 0f32;
                let element_spans_complete_output = element_geometry.contains_rect(output_geometry);
                let overlaps_with_underlay = self
                    .planes
                    .overlay
                    .iter()
                    .filter(|p| {
                        p.zpos.unwrap_or_default() < self.surface.plane_info().zpos.unwrap_or_default()
                    })
                    .any(|p| next_frame_state.overlaps(p.handle, element_geometry));
                !overlaps_with_underlay
                    && (crtc_background_matches_clear_color
                        || (element_spans_complete_output && element_is_opaque))
            } else {
                false
            };

            match self.try_assign_element(
                renderer,
                *element,
                index,
                element_geometry,
                element_is_opaque,
                &mut element_states,
                &primary_plane_elements,
                output_scale,
                &mut next_frame_state,
                output_transform,
                output_geometry,
                try_assign_primary_plane,
                frame_flags,
            ) {
                Ok(direct_scan_out_plane) => {
                    match direct_scan_out_plane.type_ {
                        drm::control::PlaneType::Overlay => {
                            overlay_plane_elements.insert(direct_scan_out_plane.handle, element);
                        }
                        drm::control::PlaneType::Primary => primary_plane_scanout_element = Some(element),
                        drm::control::PlaneType::Cursor => cursor_plane_element = Some(element),
                    }

                    if let Some(state) = render_element_states.states.get_mut(element_id) {
                        state.presentation_state = RenderElementPresentationState::ZeroCopy;
                        state.visible_area += element_visible_area;
                    } else {
                        render_element_states.states.insert(
                            element_id.clone(),
                            RenderElementState::zero_copy(*element_visible_area),
                        );
                    }
                }
                Err(reason) => {
                    if let Some(reason) = reason {
                        if !render_element_states.states.contains_key(element_id) {
                            render_element_states.states.insert(
                                element_id.clone(),
                                RenderElementState::rendering_with_reason(reason),
                            );
                        }
                    }

                    primary_plane_elements.push(element);
                }
            }
        }

        // Cleanup old state (e.g. old dmabuffers)
        for element_state in element_states.values_mut() {
            element_state.fb_cache.cleanup();
        }
        self.element_states = element_states;
        self.previous_element_states.clear();
        opaque_regions.clear();
        self.opaque_regions = opaque_regions;

        let previous_state = self
            .pending_frame
            .as_ref()
            .map(|pending| &pending.frame)
            .unwrap_or(&self.current_frame);

        // Check if the next frame state is fully compatible with the previous frame state.
        // If not do a single atomic commit test and when that fails render everything that failed
        // the test on the primary plane. This will also automatically correct any mistake we made
        // during plane assignment and start the full test cycle on the next frame.
        if next_frame_state
            .test_state_complete(
                previous_state,
                &self.surface,
                self.supports_fencing,
                false,
                allow_partial_update,
            )
            .is_err()
        {
            trace!("atomic test failed for frame, resetting frame");

            let mut removed_overlay_elements: Vec<(usize, &E)> = Vec::with_capacity(
                next_frame_state
                    .planes
                    .iter()
                    .filter(|(_, state)| state.needs_test)
                    .count(),
            );
            for (plane, state) in next_frame_state.planes.iter_mut() {
                // We can skip everything that is known to work already
                if !state.needs_test {
                    continue;
                }

                // Check if the element we are potentially going to remove is
                // on the primary plane, cursor plane or an overlay plane
                let element = if *plane == self.surface.plane() {
                    primary_plane_scanout_element.take()
                } else if self.planes.cursor.iter().any(|p| *plane == p.handle) {
                    cursor_plane_element.take()
                } else {
                    overlay_plane_elements.shift_remove(plane)
                };

                // If we have no element on this plane skip the rest
                let Some(element) = element else {
                    continue;
                };

                // Reset the plane config and state
                state.config = None;
                state.skip = false;
                state.needs_test = false;
                let element_z_index = state.element_state.take().map(|s| s.z_index).unwrap_or_default();
                removed_overlay_elements.push((element_z_index, element));
                // Note: This might not be completely correct if the same element is present
                // multiple times and only gets removed once. But this is pretty unlikely to
                // happen and will only result in reporting wrong visible area size and scan-out state
                // for a single frame.
                render_element_states.states.remove(element.id());
            }

            // If we removed any element from some plane we have
            // to make sure we actually have a slot on the primary
            // plane we can render into
            if !removed_overlay_elements.is_empty() {
                next_frame_state.set_state(self.surface.plane(), primary_plane_state);
            }

            removed_overlay_elements.sort_by_key(|(z_index, _)| *z_index);
            primary_plane_elements = removed_overlay_elements
                .into_iter()
                .map(|(_, element)| element)
                .chain(primary_plane_elements.into_iter())
                .collect();
        }

        // If a plane has been moved or no longer has a buffer we need to report that as damage
        for (handle, previous_plane_state) in previous_state.planes.iter() {
            // plane has been removed, so remove the plane from the plane id cache
            if previous_plane_state.config.is_some()
                && next_frame_state
                    .plane_state(*handle)
                    .as_ref()
                    .and_then(|state| state.config.as_ref())
                    .is_none()
            {
                self.overlay_plane_element_ids.remove_plane(handle);
            }
        }

        let render = next_frame_state
            .plane_buffer(self.surface.plane())
            .map(|config| matches!(config.buffer, ScanoutBuffer::Swapchain(_)))
            .unwrap_or(false);

        if render {
            trace!(
                "rendering {} elements on the primary {:?}",
                primary_plane_elements.len(),
                self.surface.plane(),
            );
            let (mut dmabuf, age) = {
                let primary_plane_state = next_frame_state.plane_state(self.surface.plane()).unwrap();
                let config = primary_plane_state.config.as_ref().unwrap();
                let slot = match &config.buffer.buffer {
                    ScanoutBuffer::Swapchain(slot) => slot,
                    _ => unreachable!(),
                };

                // It is safe to call export multiple times as the Slot will cache the dmabuf for us
                let dmabuf = slot.export().map_err(FrameError::AsDmabufError)?;
                let age = slot.age().into();
                (dmabuf, age)
            };

            // store the current renderer debug flags and replace them
            // with our own
            let renderer_debug_flags = renderer.debug_flags();
            renderer.set_debug_flags(self.debug_flags);

            // First we collect all our fake elements for overlay and underlays
            // This is used to transport the opaque regions for elements that
            // have been assigned to planes and to realize hole punching for
            // underlays. We use an Id per plane/element combination to not
            // interfere with the element damage state in the output damage tracker.
            // Using the original element id would store the commit in the
            // OutputDamageTracker without actual rendering anything -> bad
            // Using a id per plane could result in an issue when a different
            // element with the same geometry gets assigned and has the same
            // commit -> unlikely but possible
            // So we use an Id per plane for as long as we have the same element
            // on that plane.
            let overlay_plane_elements = overlay_plane_elements.iter().filter_map(|(p, element)| {
                let id = self
                    .overlay_plane_element_ids
                    .plane_id_for_element_id(p, element.id());

                let plane_z_pos = self
                    .planes
                    .overlay
                    .iter()
                    .find_map(|info| {
                        if info.handle == *p {
                            Some(info.zpos.unwrap_or_default())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default();
                let is_underlay = plane_z_pos < self.surface.plane_info().zpos.unwrap_or_default();
                if is_underlay {
                    Some(HolepunchRenderElement::from_render_element(id, element, output_scale).into())
                } else {
                    OverlayPlaneElement::from_render_element(id, *element, output_scale)
                        .map(DrmRenderElements::from)
                }
            });
            // Then render all remaining elements assigned to the primary plane
            let elements = overlay_plane_elements
                .chain(
                    primary_plane_elements
                        .into_iter()
                        .map(|e| DrmRenderElements::Other(e)),
                )
                .collect::<Vec<_>>();

            let mut framebuffer = renderer
                .bind(&mut dmabuf)
                .map_err(|err| RenderFrameError::RenderFrame(OutputDamageTrackerError::Rendering(err)))?;
            let render_res =
                self.damage_tracker
                    .render_output(renderer, &mut framebuffer, age, &elements, clear_color);

            // restore the renderer debug flags
            renderer.set_debug_flags(renderer_debug_flags);

            match render_res {
                Ok(render_output_result) => {
                    if render_output_result.damage.is_none() {
                        // if we receive no damage we can assume no rendering took place
                        // and we should trigger a cleanup of the renderer texture cache
                        // to prevent holding textures longer then necessary
                        let _ = renderer.cleanup_texture_cache();
                    }

                    for (id, state) in render_output_result.states.states.into_iter() {
                        // Skip the state for our fake elements
                        if self.overlay_plane_element_ids.contains_plane_id(&id) {
                            continue;
                        }

                        if let Some(existing_state) = render_element_states.states.get_mut(&id) {
                            if matches!(
                                existing_state.presentation_state,
                                RenderElementPresentationState::Skipped
                            ) {
                                *existing_state = state;
                            } else {
                                existing_state.visible_area += state.visible_area;
                            }
                        } else {
                            render_element_states.states.insert(id.clone(), state);
                        }
                    }

                    // Fixup damage on plane, if we used the plane for direct scan-out before
                    // but now use it for rendering we do not replace the damage which is
                    // the whole plane initially.
                    let had_direct_scan_out = previous_state
                        .plane_state(self.surface.plane())
                        .map(|state| state.element_state.is_some())
                        .unwrap_or(true);

                    let primary_plane_state = next_frame_state.plane_state_mut(self.surface.plane()).unwrap();
                    let config = primary_plane_state.config.as_mut().unwrap();

                    if !had_direct_scan_out {
                        if let Some(render_damage) = render_output_result.damage {
                            trace!("rendering damage: {:?}", render_damage);

                            self.primary_plane_damage_bag.add(render_damage.iter().map(|d| {
                                d.to_logical(1).to_buffer(
                                    1,
                                    Transform::Normal,
                                    &output_geometry.size.to_logical(1),
                                )
                            }));

                            // Here we need to apply the output_transform to the damage since we
                            // haven't rotated our framebuffer. dst is already in the same
                            // coordinate space src is, so no transform needed.
                            config.damage_clips = PlaneDamageClips::from_damage(
                                self.surface.device_fd(),
                                config.properties.src,
                                config.properties.dst,
                                Transform::Normal,
                                output_transform,
                                render_damage.iter().copied(),
                            )
                            .ok()
                            .flatten();
                            config.sync = Some((render_output_result.sync.clone(), None));
                        } else {
                            trace!("skipping primary plane, no damage");

                            primary_plane_state.skip = true;
                            *config = previous_state
                                .plane_state(self.surface.plane())
                                .and_then(|state| state.config.as_ref().cloned())
                                .unwrap_or_else(|| config.clone());
                        }
                    } else {
                        trace!(
                            "clearing previous direct scan-out on primary plane, damaging complete output"
                        );
                        self.primary_plane_damage_bag
                            .add([output_geometry.to_logical(1).to_buffer(
                                1,
                                Transform::Normal,
                                &output_geometry.size.to_logical(1),
                            )]);

                        config.sync = Some((render_output_result.sync.clone(), None));
                    }
                }
                Err(err) => {
                    // Rendering failed at some point, reset the buffers
                    // as we probably now have some half drawn buffer
                    self.swapchain.reset_buffers();
                    return Err(RenderFrameError::from(err));
                }
            }
        } else {
            // if we are constantly doing direct scan-out on the primary plane
            // we have to cleanup the renderer texture cache as this would
            // only happen implicit during rendering otherwise
            let _ = renderer.cleanup_texture_cache();
        }

        let primary_plane_element = if render {
            let (slot, sync) = {
                let primary_plane_state = next_frame_state.plane_state(self.surface.plane()).unwrap();
                let config = primary_plane_state.config.as_ref().unwrap();
                (
                    config.buffer.clone(),
                    config
                        .sync
                        .as_ref()
                        .map(|(sync, _)| sync.clone())
                        .unwrap_or_default(),
                )
            };

            PrimaryPlaneElement::Swapchain(PrimarySwapchainElement {
                slot,
                transform: output_transform,
                damage: self.primary_plane_damage_bag.snapshot(),
                sync,
            })
        } else {
            PrimaryPlaneElement::Element(primary_plane_scanout_element.unwrap())
        };

        // if the update only contains a cursor position update, skip it for vrr
        if frame_flags.contains(FrameFlags::SKIP_CURSOR_ONLY_UPDATES)
            && allow_partial_update
            && next_frame_state.planes.iter().all(|(plane, state)| {
                state.skip
                    || (self.planes.cursor.iter().any(|p| *plane == p.handle)
                        && state.buffer().map(|b| &b.fb)
                            == previous_state.plane_buffer(*plane).map(|b| &b.fb))
            })
        {
            for plane in self.planes.cursor.iter() {
                let Some(state) = next_frame_state.plane_state_mut(plane.handle) else {
                    continue;
                };
                state.skip = true;
            }
        }

        let next_frame = PreparedFrame {
            kind: if allow_partial_update {
                PreparedFrameKind::Partial
            } else {
                PreparedFrameKind::Full
            },
            frame: next_frame_state,
            async_page_flip,
            async_composited,
        };
        let frame_reference: RenderFrameResult<'a, A::Buffer, F::Framebuffer, E> = RenderFrameResult {
            is_empty: next_frame.is_empty(),
            primary_element: primary_plane_element,
            overlay_elements: overlay_plane_elements.into_values().collect(),
            cursor_element: cursor_plane_element,
            states: render_element_states,
            primary_plane_element_id: self.primary_plane_element_id.clone(),
            supports_fencing: self.supports_fencing,
        };

        // We only store the next frame if it actually contains any changes or if a commit is pending
        // Storing the (empty) frame could keep a reference to wayland buffers which
        // could otherwise be potentially released on `frame_submitted`
        if !next_frame.is_empty() {
            self.next_frame = Some(next_frame);
        }

        Ok(frame_reference)
    }

    /// Queues the current frame for scan-out.
    ///
    /// If `render_frame` has not been called prior to this function or returned no damage
    /// this function will return [`FrameError::EmptyFrame`]. Instead of calling `queue_frame` it
    /// is the callers responsibility to re-schedule the frame. A simple strategy for frame
    /// re-scheduling is to queue a one-shot timer that will trigger after approximately one
    /// retrace duration.
    ///
    /// *Note*: It is your responsibility to synchronize rendering if the [`RenderFrameResult`]
    /// returned by the previous [`render_frame`](DrmCompositor::render_frame) call returns `true` on [`RenderFrameResult::needs_sync`].
    ///
    /// *Note*: This function needs to be followed up with [`DrmCompositor::frame_submitted`]
    /// when a vblank event is received, that denotes successful scan-out of the frame.
    /// Otherwise the underlying swapchain will eventually run out of buffers.
    ///
    /// `user_data` can be used to attach some data to a specific buffer and later retrieved with [`DrmCompositor::frame_submitted`]
    #[profiling::function]
    pub fn queue_frame(&mut self, user_data: U) -> FrameResult<(), A, F> {
        if !self.surface.is_active() {
            return Err(FrameErrorType::<A, F>::DrmError(DrmError::DeviceInactive));
        }

        let prepared_frame = self.next_frame.take().ok_or(FrameErrorType::<A, F>::EmptyFrame)?;
        if prepared_frame.is_empty() {
            return Err(FrameErrorType::<A, F>::EmptyFrame);
        }

        if let Some(plane_state) = prepared_frame.frame.plane_state(self.surface.plane()) {
            if !plane_state.skip {
                let slot = plane_state.buffer().and_then(|config| match &config.buffer {
                    ScanoutBuffer::Swapchain(slot) => Some(slot),
                    _ => None,
                });

                if let Some(slot) = slot {
                    self.swapchain.submitted(slot);
                }
            }
        }

        self.queued_frame = Some(QueuedFrame {
            prepared_frame,
            user_data,
        });
        if self.pending_frame.is_none() {
            self.submit()?;
        }
        Ok(())
    }

    /// Commits the current frame for scan-out.
    ///
    /// If `render_frame` has not been called prior to this function or returned no damage
    /// this function will return [`FrameError::EmptyFrame`]. Instead of calling `commit_frame` it
    /// is the callers responsibility to re-schedule the frame. A simple strategy for frame
    /// re-scheduling is to queue a one-shot timer that will trigger after approximately one
    /// retrace duration.
    ///
    /// *Note*: It is your responsibility to synchronize rendering if the [`RenderFrameResult`]
    /// returned by the previous [`render_frame`](DrmCompositor::render_frame) call returns `true` on [`RenderFrameResult::needs_sync`].
    ///
    /// *Note*: This function should not be followed up with [`DrmCompositor::frame_submitted`]
    /// and will not generate a vblank event on the underlying device.
    pub fn commit_frame(&mut self) -> FrameResult<(), A, F> {
        if !self.surface.is_active() {
            return Err(FrameErrorType::<A, F>::DrmError(DrmError::DeviceInactive));
        }

        let mut prepared_frame = self.next_frame.take().ok_or(FrameErrorType::<A, F>::EmptyFrame)?;
        if prepared_frame.is_empty() {
            return Err(FrameErrorType::<A, F>::EmptyFrame);
        }

        if let Some(plane_state) = prepared_frame.frame.plane_state(self.surface.plane()) {
            if !plane_state.skip {
                let slot = plane_state.buffer().and_then(|config| match &config.buffer {
                    ScanoutBuffer::Swapchain(slot) => Some(slot),
                    _ => None,
                });

                if let Some(slot) = slot {
                    self.swapchain.submitted(slot);
                }
            }
        }

        let flip = prepared_frame
            .frame
            .commit(&self.surface, self.supports_fencing, false, false);

        let res = self.handle_flip(&prepared_frame, flip);

        if res.is_ok() {
            self.queued_frame = None;
            self.pending_frame = None;
            self.current_frame = prepared_frame.frame;
        }

        res
    }

    /// Re-evaluates the current state of the crtc and forces calls to [`render_frame`](DrmCompositor::render_frame)
    /// to return `false` for [`RenderFrameResult::is_empty`] until a frame is queued with [`queue_frame`](DrmCompositor::queue_frame).
    ///
    /// It is recommended to call this function after this used [`Session`](crate::backend::session::Session)
    /// gets re-activated / VT switched to.
    ///
    /// Usually you do not need to call this in other circumstances, but if
    /// the state of the crtc is modified elsewhere, you may call this function
    /// to reset it's internal state.
    pub fn reset_state(&mut self) -> Result<(), DrmError> {
        self.surface.reset_state()?;
        self.reset_pending = true;
        Ok(())
    }

    #[profiling::function]
    fn submit(&mut self) -> FrameResult<(), A, F> {
        let QueuedFrame {
            mut prepared_frame,
            user_data,
        } = self.queued_frame.take().unwrap();

        let allow_partial_update = prepared_frame.kind == PreparedFrameKind::Partial;
        let FlipOutcome {
            result: flip,
            presentation_mode,
            async_refused,
        } = if self.surface.commit_pending() {
            FlipOutcome {
                result: prepared_frame.frame.commit(
                    &self.surface,
                    self.supports_fencing,
                    allow_partial_update,
                    true,
                ),
                presentation_mode: PresentationMode::VSync,
                async_refused: false,
            }
        } else {
            let try_async = prepared_frame.async_page_flip
                && match self.async_flip_blocker(&prepared_frame) {
                    Some(blocker) => {
                        trace!(?blocker, "no async page flip, flipping at the next vblank");
                        false
                    }
                    None => true,
                };
            let (surface, supports_fencing) = (&self.surface, self.supports_fencing);
            let frame = &mut prepared_frame.frame;
            flip_frame(try_async, |mode| match mode {
                PresentationMode::Async => frame.page_flip_async(surface, surface.plane()),
                PresentationMode::VSync => {
                    frame.page_flip(surface, supports_fencing, allow_partial_update, true)
                }
            })
        };

        let res = self.handle_flip(&prepared_frame, flip);

        if res.is_ok() {
            self.pending_frame = Some(PendingFrame {
                frame: prepared_frame.frame,
                user_data,
                presentation_mode,
                async_refused,
            });
        }

        res
    }

    fn async_flip_blocker(&self, prepared_frame: &PreparedFrame<A, F>) -> Option<AsyncFlipBlocker> {
        let surface = AsyncFlipSurface {
            supported: self.surface.supports_async_page_flip(),
            full_commit: prepared_frame.kind == PreparedFrameKind::Full,
            vrr_commit_pending: self.surface.vrr_commit_pending(),
            composited: prepared_frame.async_composited,
        };
        let on_screen = self
            .pending_frame
            .as_ref()
            .map(|pending| &pending.frame)
            .unwrap_or(&self.current_frame);
        async_flip_blocker(
            surface,
            self.surface.plane(),
            prepared_frame.frame.async_flip_planes(on_screen),
        )
    }

    fn handle_flip(
        &mut self,
        prepared_frame: &PreparedFrame<A, F>,
        flip: Result<(), crate::backend::drm::error::Error>,
    ) -> FrameResult<(), A, F> {
        match flip {
            Ok(_) => {
                if prepared_frame.kind == PreparedFrameKind::Full {
                    self.reset_pending = false;
                }
            }
            Err(crate::backend::drm::error::Error::Access(ref access))
                if access.source.kind() == ErrorKind::InvalidInput =>
            {
                // In case the commit/flip failed while we tried to directly scan-out
                // something on the primary plane we can try to mark this as failed for
                // the next call to render_frame
                let primary_plane_element_state = prepared_frame
                    .frame
                    .plane_state(self.surface.plane())
                    .and_then(|plane_state| {
                        plane_state
                            .element_state
                            .as_ref()
                            .map(|element_state| &element_state.id)
                    })
                    .and_then(|primary_plane_element_id| {
                        self.element_states.get_mut(primary_plane_element_id)
                    });

                if let Some(primary_plane_element_state) = primary_plane_element_state {
                    for instance in primary_plane_element_state.instances.iter_mut() {
                        instance.failed_planes.primary = true;
                    }
                }
            }
            Err(_) => {}
        };

        flip.map_err(FrameError::DrmError)
    }

    /// Returns the frame that was committed and waits for its page flip, if any.
    ///
    /// After [`queue_frame`](DrmCompositor::queue_frame) (or a [`frame_submitted`](DrmCompositor::frame_submitted)
    /// that submitted a queued frame) this tells how that frame went out, see
    /// [`FrameFlags::ALLOW_ASYNC_PAGE_FLIP`]. Read it before `frame_submitted` to know how the
    /// frame whose page flip event just arrived was presented.
    ///
    /// The event of an async page flip arrives and has to be handled like any other, but the
    /// kernel stamps it with the time and sequence of the last vertical blank, not of the flip
    /// (see [`DrmSurface::page_flip_async`]).
    pub fn pending_frame(&self) -> Option<FrameRef<'_, U>> {
        self.pending_frame.as_ref().map(|frame| FrameRef {
            user_data: &frame.user_data,
            presentation_mode: frame.presentation_mode,
            async_refused: frame.async_refused,
        })
    }

    /// Marks the current frame as submitted.
    ///
    /// *Note*: Needs to be called, after the vblank event of the matching [`DrmDevice`](super::DrmDevice)
    /// was received after calling [`DrmCompositor::queue_frame`] on this surface.
    /// Otherwise the underlying swapchain will run out of buffers eventually.
    #[profiling::function]
    pub fn frame_submitted(&mut self) -> FrameResult<Option<U>, A, F> {
        if let Some(PendingFrame {
            mut frame, user_data, ..
        }) = self.pending_frame.take()
        {
            std::mem::swap(&mut frame, &mut self.current_frame);
            if self.queued_frame.is_some() {
                self.submit()?;
            }
            Ok(Some(user_data))
        } else {
            Ok(None)
        }
    }

    /// Reset the underlying buffers
    pub fn reset_buffers(&mut self) {
        self.swapchain.reset_buffers();
    }

    /// Reset the age for all buffers.
    ///
    /// This can be used to efficiently clear the damage history without having to
    /// modify the damage for each surface.
    pub fn reset_buffer_ages(&mut self) {
        self.swapchain.reset_buffer_ages();
    }

    /// Returns the underlying [`crtc`] of this surface
    pub fn crtc(&self) -> crtc::Handle {
        self.surface.crtc()
    }

    /// Returns the underlying [`plane`] of this surface
    pub fn plane(&self) -> plane::Handle {
        self.surface.plane()
    }

    /// Currently used [`connector`]s of this `Surface`
    pub fn current_connectors(&self) -> impl IntoIterator<Item = connector::Handle> {
        self.surface.current_connectors()
    }

    /// Returns the pending [`connector`]s
    /// used for the next frame queued via [`queue_frame`](DrmCompositor::queue_frame).
    pub fn pending_connectors(&self) -> impl IntoIterator<Item = connector::Handle> {
        self.surface.pending_connectors()
    }

    /// Tries to add a new [`connector`]
    /// to be used after the next commit.
    ///
    /// **Warning**: You need to make sure, that the connector is not used with another surface
    /// or was properly removed via `remove_connector` + `commit` before adding it to another surface.
    /// Behavior if failing to do so is undefined, but might result in rendering errors or the connector
    /// getting removed from the other surface without updating it's internal state.
    ///
    /// Fails if the `connector` is not compatible with the underlying [`crtc`]
    /// (e.g. no suitable [`encoder`](drm::control::encoder) may be found)
    /// or is not compatible with the currently pending
    /// [`Mode`].
    pub fn add_connector(&self, connector: connector::Handle) -> FrameResult<(), A, F> {
        self.surface
            .add_connector(connector)
            .map_err(FrameError::DrmError)
    }

    /// Tries to mark a [`connector`]
    /// for removal on the next commit.
    pub fn remove_connector(&self, connector: connector::Handle) -> FrameResult<(), A, F> {
        self.surface
            .remove_connector(connector)
            .map_err(FrameError::DrmError)
    }

    /// Tries to replace the current connector set with the newly provided one on the next commit.
    ///
    /// Fails if one new `connector` is not compatible with the underlying [`crtc`]
    /// (e.g. no suitable [`encoder`](drm::control::encoder) may be found)
    /// or is not compatible with the currently pending
    /// [`Mode`].
    pub fn set_connectors(&self, connectors: &[connector::Handle]) -> FrameResult<(), A, F> {
        self.surface
            .set_connectors(connectors)
            .map_err(FrameError::DrmError)
    }

    /// Returns the currently active [`Mode`]
    /// of the underlying [`crtc`]
    pub fn current_mode(&self) -> Mode {
        self.surface.current_mode()
    }

    /// Returns the currently pending [`Mode`]
    /// to be used after the next commit.
    pub fn pending_mode(&self) -> Mode {
        self.surface.pending_mode()
    }

    /// Tries to set a new [`Mode`]
    /// to be used after the next commit.
    ///
    /// Fails if the mode is not compatible with the underlying
    /// [`crtc`] or any of the
    /// pending [`connector`]s.
    pub fn use_mode(&mut self, mode: Mode) -> FrameResult<(), A, F> {
        self.surface.use_mode(mode).map_err(FrameError::DrmError)?;
        let (w, h) = mode.size();
        self.swapchain.resize(w as _, h as _);
        Ok(())
    }

    /// Returns if Variable Refresh Rate is advertised as supported by the given connector.
    ///
    /// See [`DrmSurface::vrr_supported`] for more details.
    pub fn vrr_supported(&self, conn: connector::Handle) -> FrameResult<VrrSupport, A, F> {
        self.surface.vrr_supported(conn).map_err(FrameError::DrmError)
    }

    /// Returns if Variable Refresh Rate is currently enabled for frames composed by this [`DrmCompositor`].
    pub fn vrr_enabled(&self) -> bool {
        self.surface.vrr_enabled()
    }

    /// Tries to set variable refresh rate (VRR) for the next frame.
    ///
    /// Doing so might cause the next frame to trigger a modeset.
    /// Check [`DrmCompositor::vrr_supported`], which indicates if VRR can be
    /// used without a modeset on the attached connectors.
    pub fn use_vrr(&mut self, vrr: bool) -> FrameResult<(), A, F> {
        self.surface.use_vrr(vrr).map_err(FrameError::DrmError)
    }

    /// Set the [`DebugFlags`] to use
    ///
    /// Note: This will reset the primary plane swapchain if
    /// the flags differ from the current flags
    pub fn set_debug_flags(&mut self, flags: DebugFlags) {
        if self.debug_flags != flags {
            self.debug_flags = flags;
            self.swapchain.reset_buffers();
        }
    }

    /// Returns the current enabled [`DebugFlags`]
    pub fn debug_flags(&self) -> DebugFlags {
        self.debug_flags
    }

    /// Returns a reference to the underlying drm surface
    pub fn surface(&self) -> &DrmSurface {
        &self.surface
    }

    /// Get the format of the underlying swapchain
    pub fn format(&self) -> DrmFourcc {
        self.swapchain.format()
    }

    /// Get the allowed modifiers of the underlying swapchain
    pub fn modifiers(&self) -> &[DrmModifier] {
        self.swapchain.modifiers()
    }

    /// Reset the underlying swapchain and assign a new color format.
    pub fn set_format(
        &mut self,
        allocator: A,
        code: DrmFourcc,
        modifiers: impl IntoIterator<Item = DrmModifier>,
    ) -> Result<(), FrameErrorType<A, F>> {
        let (swapchain, is_oapque) = Self::test_format(
            &self.surface,
            self.supports_fencing,
            &self.planes,
            allocator,
            &self.framebuffer_exporter,
            code,
            modifiers,
        )
        .map_err(|(_, err)| err)?;

        self.swapchain = swapchain;
        self.primary_is_opaque = is_oapque;

        Ok(())
    }

    /// Change the output mode source.
    pub fn set_output_mode_source(&mut self, output_mode_source: OutputModeSource) {
        // Avoid clearing damage if mode source did not change.
        if output_mode_source == self.output_mode_source {
            return;
        }

        self.damage_tracker = OutputDamageTracker::from_mode_source(output_mode_source.clone());
        self.output_mode_source = output_mode_source;
    }

    #[allow(clippy::too_many_arguments)]
    #[instrument(level = "trace", skip_all)]
    #[profiling::function]
    fn try_assign_element<'a, R, E>(
        &mut self,
        renderer: &mut R,
        element: &'a E,
        element_zindex: usize,
        element_geometry: Rectangle<i32, Physical>,
        element_is_opaque: bool,
        element_states: &mut IndexMap<Id, ElementState<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>,
        primary_plane_elements: &[&'a E],
        scale: Scale<f64>,
        frame_state: &mut CompositorFrameState<A, F>,
        output_transform: Transform,
        output_geometry: Rectangle<i32, Physical>,
        try_assign_primary_plane: bool,
        frame_flags: FrameFlags,
    ) -> Result<PlaneAssignment, Option<RenderingReason>>
    where
        R: Renderer + Bind<Dmabuf>,
        E: RenderElement<R>,
    {
        // Check if we have a free plane, otherwise we can exit early
        if !frame_flags.intersects(FrameFlags::ALLOW_SCANOUT) {
            trace!(
                "skipping direct scan-out for element {:?}, no free planes",
                element.id()
            );
            return Err(None);
        };
        if element.is_framebuffer_effect() {
            return Err(None);
        }

        let mut rendering_reason: Option<RenderingReason> = None;

        if try_assign_primary_plane {
            match self.try_assign_primary_plane(
                renderer,
                element,
                element_zindex,
                element_geometry,
                element_states,
                scale,
                frame_state,
                output_transform,
                output_geometry,
                frame_flags,
            ) {
                Ok(plane) => {
                    trace!(
                        "assigned element {:?} to primary {:?}",
                        element.id(),
                        self.surface.plane()
                    );
                    return Ok(plane);
                }
                Err(err) => rendering_reason = rendering_reason.or(err),
            };
        }

        if let Some(plane) = self.try_assign_cursor_plane(
            renderer,
            element,
            element_zindex,
            element_geometry,
            scale,
            frame_state,
            output_transform,
            output_geometry,
            frame_flags,
        ) {
            trace!("assigned element {:?} to cursor {:?}", element.id(), plane.handle);
            return Ok(plane);
        }

        match self.try_assign_overlay_plane(
            renderer,
            element,
            element_zindex,
            element_geometry,
            element_is_opaque,
            element_states,
            primary_plane_elements,
            scale,
            frame_state,
            output_transform,
            output_geometry,
            frame_flags,
        ) {
            Ok(plane) => {
                trace!(
                    "assigned element {:?} to overlay plane {:?}",
                    element.id(),
                    plane.handle
                );
                return Ok(plane);
            }
            Err(err) => rendering_reason = rendering_reason.or(err),
        }

        Err(rendering_reason)
    }

    #[allow(clippy::too_many_arguments)]
    #[instrument(level = "trace", skip_all)]
    #[profiling::function]
    fn try_assign_primary_plane<'a, R, E>(
        &mut self,
        renderer: &mut R,
        element: &'a E,
        element_zindex: usize,
        element_geometry: Rectangle<i32, Physical>,
        element_states: &mut IndexMap<Id, ElementState<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>,
        scale: Scale<f64>,
        frame_state: &mut CompositorFrameState<A, F>,
        output_transform: Transform,
        output_geometry: Rectangle<i32, Physical>,
        frame_flags: FrameFlags,
    ) -> Result<PlaneAssignment, Option<RenderingReason>>
    where
        R: Renderer,
        E: RenderElement<R>,
    {
        if !frame_flags
            .intersects(FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT | FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT_ANY)
        {
            return Err(None);
        }

        if frame_state
            .plane_state(self.surface.plane())
            .map(|state| state.element_state.is_some())
            .unwrap_or(true)
        {
            return Err(None);
        }

        let element_config = self.element_config(
            renderer,
            element,
            element_zindex,
            element_geometry,
            element_states,
            frame_state,
            output_transform,
            output_geometry,
            true,
        )?;

        if let ScanoutBuffer::Swapchain(slot) = &frame_state
            .plane_buffer(self.surface.plane())
            .expect("We have a buffer for the primary plane")
            .buffer
        {
            if !frame_flags.contains(FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT_ANY)
                && slot.format() != element_config.properties.format
            {
                trace!(
                    "failed to assign element {:?} to primary {:?}, format doesn't match",
                    element.id(),
                    self.surface.plane()
                );
                return Err(None);
            }
        }

        let has_underlay = self
            .planes
            .overlay
            .iter()
            .filter(|plane| {
                self.surface.plane_info().zpos.unwrap_or_default() > plane.zpos.unwrap_or_default()
            })
            .any(|plane| frame_state.is_assigned(plane.handle));

        if has_underlay {
            trace!(
                "failed to assign element {:?} to primary {:?}, already has underlay",
                element.id(),
                self.surface.plane()
            );
            return Err(None);
        }

        if element_config.failed_planes.primary {
            return Err(Some(RenderingReason::ScanoutFailed));
        }

        let res = self.try_assign_plane(
            element,
            &element_config,
            self.surface.plane_info(),
            scale,
            frame_state,
        );

        if let Err(Some(RenderingReason::ScanoutFailed)) = res {
            element_config.failed_planes.primary = true;
        }

        res
    }

    #[allow(clippy::too_many_arguments)]
    #[instrument(level = "trace", skip_all)]
    #[profiling::function]
    fn try_assign_cursor_plane<R, E>(
        &mut self,
        renderer: &mut R,
        element: &E,
        element_zindex: usize,
        element_geometry: Rectangle<i32, Physical>,
        scale: Scale<f64>,
        frame_state: &mut CompositorFrameState<A, F>,
        output_transform: Transform,
        output_geometry: Rectangle<i32, Physical>,
        frame_flags: FrameFlags,
    ) -> Option<PlaneAssignment>
    where
        R: Renderer,
        E: RenderElement<R>,
    {
        if !frame_flags.contains(FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT) {
            return None;
        }

        let Some(cursor_state) = self.cursor_state.as_mut() else {
            trace!("no cursor state, skipping cursor rendering");
            return None;
        };

        // only try to assign elements on a cursor plane that indicate so
        if element.kind() != Kind::Cursor {
            trace!(
                "skipping element {:?} on cursor plane(s), element kind not cursor",
                element.id(),
            );
            return None;
        }

        let element_size = output_transform.transform_size(element_geometry.size);

        // if the element is greater than the cursor size we can not
        // use the cursor plane to scan out the element
        if element_size.w > self.cursor_size.w || element_size.h > self.cursor_size.h {
            trace!("element {:?} too big for cursor plane(s), skipping", element.id(),);
            return None;
        }

        // For now we only support a single cursor plane, so first test if we already
        // assigned something to any cursor plane
        if let Some(plane_info) = self
            .planes
            .cursor
            .iter()
            .find(|plane_info| frame_state.is_assigned(plane_info.handle))
        {
            trace!(
                "skipping element {:?} on cursor {:?}, plane already has element assigned",
                element.id(),
                plane_info.handle
            );
            return None;
        }

        let previous_state = self
            .pending_frame
            .as_ref()
            .map(|pending| &pending.frame)
            .unwrap_or(&self.current_frame);

        // In case we have multiple cursor planes we will try to keep using
        // the same cursor plane for as long as possible.
        // So first we test the previous state for an assigned cursor plane and
        // if that fails we will try to pick the first unclaimed cursor plane.
        let Some((plane_info, plane_claim)) = self
            .planes
            .cursor
            .iter()
            .find_map(|plane_info: &PlaneInfo| {
                if previous_state.is_assigned(plane_info.handle) {
                    self.surface
                        .claim_plane(plane_info.handle)
                        .map(|claim| (plane_info, claim))
                } else {
                    None
                }
            })
            .or_else(|| {
                self.planes.cursor.iter().find_map(|plane_info| {
                    self.surface
                        .claim_plane(plane_info.handle)
                        .map(|claim| (plane_info, claim))
                })
            })
        else {
            trace!(
                "skipping element {:?} on cursor plane(s), no free plane found",
                element.id(),
            );
            return None;
        };

        let cursor_plane_size = if let Some(size_hints) = plane_info.size_hints.as_deref() {
            // size hints are in order of preference, so we can choose the first one
            // that can hold the whole element
            //
            // Note: we use the legacy cursor size as a pre-check and expect it to hold the
            // biggest possible size
            size_hints
                .iter()
                .find(|hint| hint.w as i32 >= element_size.w && hint.h as i32 >= element_size.h)
                .map(|hint| Size::<i32, Physical>::from((hint.w as i32, hint.h as i32)))
                .unwrap_or(self.cursor_size)
        } else {
            self.cursor_size
        };

        // this calculates the location of the cursor plane taking the simulated transform
        // into consideration
        let cursor_plane_location = output_transform
            .transform_point_in(element.location(scale), &output_geometry.size)
            - output_transform.transform_point_in(Point::default(), &cursor_plane_size);

        let previous_state = self
            .pending_frame
            .as_ref()
            .map(|pending| &pending.frame)
            .unwrap_or(&self.current_frame);

        let previous_element_state = previous_state
            .plane_state(plane_info.handle)
            .and_then(|state| state.element_state.as_ref());

        // if the output transform or scale change we have to (re-)render the cursor plane,
        // also if the element changed or reports damage we have to render it
        let render = cursor_state
            .previous_output_transform
            .map(|t| t != output_transform)
            .unwrap_or(true)
            || cursor_state
                .previous_output_scale
                .map(|s| s != scale)
                .unwrap_or(true)
            || previous_element_state
                .map(|element_state| {
                    element_state.id != *element.id()
                        || element.current_commit() != element_state.commit
                        || element_state.cursor_size != Some(element_size)
                })
                .unwrap_or(true)
            || previous_state
                .plane_state(plane_info.handle)
                .and_then(|state| {
                    state
                        .config
                        .as_ref()
                        .map(|config| config.properties.dst.size != cursor_plane_size)
                })
                .unwrap_or(true);

        // check if the cursor plane location changed
        let reposition = previous_state
            .plane_state(plane_info.handle)
            .and_then(|state| {
                state
                    .config
                    .as_ref()
                    .map(|config| config.properties.dst.loc != cursor_plane_location)
            })
            .unwrap_or(true);

        // ok, nothing changed, try to keep the previous state
        if !render && !reposition {
            let mut plane_state = previous_state.plane_state(plane_info.handle).unwrap().clone();
            plane_state.skip = true;
            // Note: we know that we had a cursor plane in the
            // previous frame and that nothing changed. In this
            // case skip the whole testing
            plane_state.needs_test = false;
            frame_state.set_state(plane_info.handle, plane_state);
            return Some(plane_info.into());
        }

        // we no not have to re-render but update the planes location
        if !render && reposition {
            trace!("repositioning cursor plane");
            let mut plane_state = previous_state.plane_state(plane_info.handle).unwrap().clone();
            plane_state.skip = false;
            // Note: we know that we had a cursor plane in the
            // previous frame, so we assume a simple location change
            // does not not to be tested
            plane_state.needs_test = false;
            let config = plane_state.config.as_mut().unwrap();
            config.properties.dst.loc = cursor_plane_location;
            frame_state.set_state(plane_info.handle, plane_state);
            return Some(plane_info.into());
        }

        trace!(
            "trying to render element {:?} on cursor {:?}",
            element.id(),
            plane_info.handle
        );

        // if we fail to create a buffer we can just return false and
        // force the cursor to be rendered on the primary plane
        let mut cursor_buffer = match cursor_state.allocator.create_buffer(
            cursor_plane_size.w as u32,
            cursor_plane_size.h as u32,
            DrmFourcc::Argb8888,
            &[DrmModifier::Linear],
        ) {
            Ok(buffer) => buffer,
            Err(err) => {
                debug!("failed to create cursor buffer: {}", err);
                return None;
            }
        };

        // if we fail to export a framebuffer for our buffer we can skip the rest
        let framebuffer = match cursor_state.framebuffer_exporter.add_framebuffer(
            self.surface.device_fd(),
            ExportBuffer::Allocator(&cursor_buffer),
            false,
        ) {
            Ok(Some(fb)) => fb,
            Ok(None) => {
                debug!(
                    "failed to export framebuffer for cursor {:?}: no framebuffer available",
                    plane_info.handle
                );
                return None;
            }
            Err(err) => {
                debug!(
                    "failed to export framebuffer for cursor {:?}: {}",
                    plane_info.handle, err
                );
                return None;
            }
        };

        let cursor_buffer_size = cursor_plane_size.to_logical(1).to_buffer(1, Transform::Normal);

        #[cfg(not(feature = "renderer_pixman"))]
        if !copy_element_to_cursor_bo(
            renderer,
            element,
            element_size,
            cursor_plane_size,
            output_transform,
            &mut cursor_buffer,
        ) {
            tracing::trace!("failed to copy element to cursor bo, skipping element on cursor plane");
            return None;
        }

        #[cfg(feature = "renderer_pixman")]
        if !copy_element_to_cursor_bo(
            renderer,
            element,
            element_size,
            cursor_plane_size,
            output_transform,
            &mut cursor_buffer,
        ) {
            profiling::scope!("render cursor plane");
            tracing::trace!("cursor fast-path copy failed, falling back to rendering using offscreen buffer");

            let Some(storage) = element.underlying_storage(renderer) else {
                trace!("Can't obtain cursor's underlying storage");
                return None;
            };

            let pixman_renderer = cursor_state.pixman_renderer.as_mut()?;

            // Create a pixman image from the source cursor data. This will either be set by the
            // client, or the compositor's choice.
            let cursor_texture = match storage {
                UnderlyingStorage::Wayland(buffer) => pixman_renderer
                    .import_buffer(buffer, None, &[element.src().to_i32_up()])
                    .transpose()
                    .ok()
                    .flatten(),
                UnderlyingStorage::Memory(memory) => {
                    let format = memory.format();
                    let size = memory.size();
                    let Ok(pixman_format) = pixman::FormatCode::try_from(format) else {
                        debug!("No pixman format for {format}");
                        return None;
                    };
                    unsafe {
                        match pixman::Image::from_raw_mut(
                            pixman_format,
                            size.w as usize,
                            size.h as usize,
                            memory.as_ptr() as *mut u32,
                            memory.stride() as usize,
                            false,
                        ) {
                            Ok(image) => Some(PixmanTexture::from(image)),
                            Err(e) => {
                                debug!("pixman cursor: {e}");
                                None
                            }
                        }
                    }
                }
            }?;

            let ret = cursor_buffer
                .map_mut::<_, Result<_, PixmanError>>(
                    0,
                    0,
                    cursor_buffer_size.w as u32,
                    cursor_buffer_size.h as u32,
                    |mbo| {
                        let plane_pixman_format = pixman::FormatCode::try_from(DrmFourcc::Argb8888).unwrap();
                        let mut cursor_dst = unsafe {
                            pixman::Image::from_raw_mut(
                                plane_pixman_format,
                                mbo.width() as usize,
                                mbo.height() as usize,
                                mbo.buffer_mut().as_mut_ptr() as *mut u32,
                                mbo.stride() as usize,
                                false,
                            )
                        }
                        .map_err(|_| PixmanError::ImportFailed)?;
                        let mut framebuffer = pixman_renderer.bind(&mut cursor_dst)?;
                        let mut frame =
                            pixman_renderer.render(&mut framebuffer, cursor_plane_size, output_transform)?;
                        frame.clear(Color32F::TRANSPARENT, &[Rectangle::from_size(cursor_plane_size)])?;
                        let src = element.src();
                        let dst = Rectangle::from_size(element_geometry.size);
                        frame.render_texture_from_to(
                            &cursor_texture,
                            src,
                            dst,
                            &[dst],
                            &[],
                            element.transform(),
                            element.alpha(),
                        )?;
                        let _ = frame.finish()?.wait(); // what can we do?
                        Ok(())
                    },
                )
                .expect("Lost track of cursor device");

            if let Err(err) = ret {
                debug!("{err}");
                return None;
            }
        };

        let src = Rectangle::from_size(cursor_buffer_size).to_f64();
        let dst = Rectangle::new(cursor_plane_location, cursor_plane_size);

        let config = PlaneConfig {
            properties: PlaneProperties {
                src,
                dst,
                alpha: 1.0,
                transform: Transform::Normal,
                format: framebuffer.format(),
            },
            buffer: DrmScanoutBuffer {
                buffer: ScanoutBuffer::Cursor(Arc::new(cursor_buffer)),
                fb: CachedDrmFramebuffer::new(DrmFramebuffer::Gbm(framebuffer)),
            },
            damage_clips: None,
            plane_claim,
            sync: None,
        };
        let is_compatible = previous_state
            .plane_state(plane_info.handle)
            .map(|state| {
                state
                    .config
                    .as_ref()
                    .map(|other| {
                        // Note: We do not use the plane config `is_compatible` test
                        // here as we exclude the destination location from the test
                        other.properties.src == config.properties.src
                            && other.properties.dst.size == config.properties.dst.size
                            && other.properties.alpha == config.properties.alpha
                            && other.properties.transform == config.properties.transform
                            && other.properties.format == config.properties.format
                    })
                    .unwrap_or(false)
            })
            .unwrap_or(false);

        let plane_state = PlaneState {
            skip: false,
            // Note: we assume we only have to test if the plane is
            // not compatible. This should only happen if we either
            // had no cursor plane before or we did direct scan-out
            // on it. A simple re-position without re-render is
            // already handled earlier.
            needs_test: !is_compatible,
            element_state: Some(PlaneElementState {
                id: element.id().clone(),
                commit: element.current_commit(),
                z_index: element_zindex,
                cursor_size: Some(element_size),
            }),
            config: Some(config),
        };

        let res = if is_compatible {
            frame_state.set_state(plane_info.handle, plane_state);
            true
        } else {
            frame_state
                .test_state(
                    &self.surface,
                    self.supports_fencing,
                    plane_info.handle,
                    plane_state,
                    false,
                )
                .is_ok()
        };

        if res {
            cursor_state.previous_output_scale = Some(scale);
            cursor_state.previous_output_transform = Some(output_transform);
            Some(plane_info.into())
        } else {
            info!("failed to test cursor {:?} state", plane_info.handle);
            None
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[instrument(level = "trace", skip_all)]
    #[profiling::function]
    fn element_config<'a, R, E>(
        &mut self,
        renderer: &mut R,
        element: &E,
        element_zindex: usize,
        element_geometry: Rectangle<i32, Physical>,
        element_states: &'a mut IndexMap<Id, ElementState<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>,
        frame_state: &mut CompositorFrameState<A, F>,
        output_transform: Transform,
        output_geometry: Rectangle<i32, Physical>,
        allow_opaque_fallback: bool,
    ) -> Result<
        ElementPlaneConfig<
            'a,
            <A as Allocator>::Buffer,
            <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer,
        >,
        ExportBufferError,
    >
    where
        R: Renderer,
        E: RenderElement<R>,
    {
        let element_id = element.id();

        // We can only try to do direct scan-out for element that provide a underlying storage
        let underlying_storage = element
            .underlying_storage(renderer)
            .ok_or(ExportBufferError::NoUnderlyingStorage)?;

        let export_buffer = ExportBuffer::from_underlying_storage(&underlying_storage)
            .ok_or(ExportBufferError::Unsupported)?;

        if !self.framebuffer_exporter.can_add_framebuffer(&export_buffer) {
            return Err(ExportBufferError::Unsupported);
        }

        // First we try to find a state in our new states, this is important if
        // we got the same id multiple times. If we can't find it we use the previous
        // state if available
        if !element_states.contains_key(element_id) {
            let previous_fb_cache = self
                .previous_element_states
                .get_mut(element_id)
                // Note: We can mem::take the old fb_cache here here as we guarantee that
                // the element state will always overwrite the current state at the end of render_frame
                .map(|state| std::mem::take(&mut state.fb_cache))
                .unwrap_or_default();
            element_states.insert(
                element_id.clone(),
                ElementState {
                    instances: SmallVec::new(),
                    fb_cache: previous_fb_cache,
                },
            );
        }
        let element_fb_cache: &mut ElementFramebufferCache<
            <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer,
        > = element_states
            .get_mut(element_id)
            .map(|state| &mut state.fb_cache)
            .unwrap();

        let element_cache_key =
            ElementFramebufferCacheKey::from_underlying_storage(&underlying_storage, allow_opaque_fallback)
                .ok_or(ExportBufferError::Unsupported)?;
        let cached_fb = element_fb_cache.get(&element_cache_key);

        if cached_fb.is_none() {
            trace!(
                "no cached fb, exporting new fb for element {:?} underlying storage {:?}",
                element_id, &underlying_storage
            );

            let fb = self
                .framebuffer_exporter
                .add_framebuffer(self.surface.device_fd(), export_buffer, allow_opaque_fallback)
                .map_err(|err| {
                    trace!("failed to add framebuffer: {:?}", err);
                    ExportBufferError::ExportFailed
                })
                .and_then(|fb| {
                    fb.map(|fb| CachedDrmFramebuffer::new(DrmFramebuffer::Exporter(fb)))
                        .ok_or(ExportBufferError::Unsupported)
                });

            if fb.is_err() {
                trace!(
                    "could not import framebuffer for element {:?} underlying storage {:?}",
                    element_id, &underlying_storage
                );
            }

            element_fb_cache.insert(element_cache_key.clone(), fb);
        } else {
            trace!(
                "using cached fb for element {:?} underlying storage {:?}",
                element_id, &underlying_storage
            );
        }

        let fb: &CachedDrmFramebuffer<<F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer> =
            element_fb_cache.get(&element_cache_key).unwrap()?;

        let src = element.src();
        let dst = output_transform.transform_rect_in(element_geometry, &output_geometry.size);
        // the output transform we are passed is already inverted to represent CW rotation (this is done to match what the
        // renderer is doing), but drm and the elements actually use/expect CCW rotation. to solve this we just invert
        // the transform again here.
        let transform = apply_output_transform(
            apply_underlying_storage_transform(element.transform(), &underlying_storage),
            output_transform.invert(),
        );
        let alpha = element.alpha();
        let properties = PlaneProperties {
            src,
            dst,
            alpha,
            transform,
            format: fb.format(),
        };
        let buffer: DrmScanoutBuffer<
            <A as Allocator>::Buffer,
            <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer,
        > = ScanoutBuffer::from_underlying_storage(underlying_storage)
            .map(|buffer| DrmScanoutBuffer {
                fb: fb.clone(),
                buffer,
            })
            .ok_or(ExportBufferError::Unsupported)?;

        if !element_states
            .get(element_id)
            .unwrap()
            .instances
            .iter()
            .any(|i| i.properties == properties)
        {
            let overlay_bitmask =
                self.planes
                    .overlay
                    .iter()
                    .enumerate()
                    .fold(0u32, |mut acc, (index, plane)| {
                        if frame_state.is_assigned(plane.handle) {
                            acc |= 1 << index;
                        }
                        acc
                    });
            let cursor_bitmask =
                self.planes
                    .cursor
                    .iter()
                    .enumerate()
                    .fold(0u32, |mut acc, (index, plane)| {
                        if frame_state.is_assigned(plane.handle) {
                            acc |= 1 << index;
                        }
                        acc
                    });
            let current_plane_snapshot = PlanesSnapshot {
                primary: frame_state.is_assigned(self.surface.plane()),
                cursor_bitmask,
                overlay_bitmask,
            };

            let element_state = element_states.get_mut(element_id).unwrap();
            element_state.instances.push(ElementInstanceState {
                properties,
                active_planes: current_plane_snapshot,
                failed_planes: Default::default(),
            });

            if let Some(previous_state) = self.previous_element_states.get(element_id) {
                // lets look if we find a previous instance with exactly the same properties.
                // if we find one we can test if nothing changed and re-use the failed tests
                let matching_instance = previous_state
                    .instances
                    .iter()
                    .find(|i| i.properties == properties);

                if let Some(matching_instance) = matching_instance {
                    if current_plane_snapshot == matching_instance.active_planes {
                        let previous_frame_state = self
                            .pending_frame
                            .as_ref()
                            .map(|pending| &pending.frame)
                            .unwrap_or(&self.current_frame);

                        // Note: we ignore the cursor plane here as this would result
                        // in constant re-tests of cursor moves and we do not expect
                        // that to influence the test state of our elements.
                        // Adding or removing cursor can influence the other planes, but
                        // is already covered in the active planes check.
                        let primary_plane_changed = if current_plane_snapshot.primary {
                            frame_state.plane_properties(self.surface.plane())
                                != previous_frame_state.plane_properties(self.surface.plane())
                        } else {
                            false
                        };

                        let overlay_plane_changed =
                            self.planes.overlay.iter().enumerate().any(|(index, plane)| {
                                // we only want to test planes that are currently in use
                                if current_plane_snapshot.overlay_bitmask & (1 << index) == 0 {
                                    return false;
                                }

                                frame_state.plane_properties(plane.handle)
                                    != previous_frame_state.plane_properties(plane.handle)
                            });

                        if !(primary_plane_changed || overlay_plane_changed) {
                            // we now know that nothing changed and we can assume any previously failed
                            // test will again fail
                            let instance_state = element_state
                                .instances
                                .iter_mut()
                                .find(|i| i.properties == properties)
                                .unwrap();
                            instance_state.failed_planes = matching_instance.failed_planes;
                        }
                    }
                }
            }
        }

        let failed_planes = element_states
            .get_mut(element_id)
            .unwrap()
            .instances
            .iter_mut()
            .find_map(|i| {
                if i.properties == properties {
                    Some(&mut i.failed_planes)
                } else {
                    None
                }
            })
            .unwrap();

        Ok(ElementPlaneConfig {
            properties,
            z_index: element_zindex,
            geometry: element_geometry,
            buffer,
            failed_planes,
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[instrument(level = "trace", skip_all)]
    #[profiling::function]
    fn try_assign_overlay_plane<'a, R, E>(
        &mut self,
        renderer: &mut R,
        element: &'a E,
        element_zindex: usize,
        element_geometry: Rectangle<i32, Physical>,
        element_is_opaque: bool,
        element_states: &mut IndexMap<Id, ElementState<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>,
        primary_plane_elements: &[&'a E],
        scale: Scale<f64>,
        frame_state: &mut CompositorFrameState<A, F>,
        output_transform: Transform,
        output_geometry: Rectangle<i32, Physical>,
        frame_flags: FrameFlags,
    ) -> Result<PlaneAssignment, Option<RenderingReason>>
    where
        R: Renderer,
        E: RenderElement<R>,
    {
        if !frame_flags.contains(FrameFlags::ALLOW_OVERLAY_PLANE_SCANOUT) {
            return Err(None);
        }

        // only try to assign elements on an overlay plane that indicate so
        if element.kind() != Kind::ScanoutCandidate && element.kind() != Kind::Cursor {
            trace!(
                "skipping element {:?} on overlay plane(s), element kind not scanout-candidate/cursor",
                element.id(),
            );
            return Err(None);
        }

        let element_id = element.id();

        // Check if we have a free plane, otherwise we can exit early
        if self
            .planes
            .overlay
            .iter()
            .all(|plane| frame_state.is_assigned(plane.handle))
        {
            trace!(
                "skipping overlay planes for element {:?}, no free planes",
                element_id
            );
            return Err(None);
        }

        let element_config = self.element_config(
            renderer,
            element,
            element_zindex,
            element_geometry,
            element_states,
            frame_state,
            output_transform,
            output_geometry,
            false,
        )?;

        let overlaps_with_primary_plane_element = primary_plane_elements.iter().any(|e| {
            let other_geometry = e.geometry(scale);
            other_geometry.overlaps(element_config.geometry)
        });

        let primary_plane_has_alpha = frame_state
            .plane_buffer(self.surface.plane())
            .map(|state| has_alpha(state.format().code))
            .unwrap_or(false);

        let previous_frame_state = self
            .pending_frame
            .as_ref()
            .map(|pending| &pending.frame)
            .unwrap_or(&self.current_frame);

        // We consider a plane compatible if the z-index of the previous assigned
        // element is or equal to our z-index and the properties (src/dst/format/...)
        // are equal. The reason for the z-index limitation is that we do not want
        // to assign ourself to the same plane if our z-index changed. That could
        // result in assigning the element on a lower plane as necessary and then
        // blocking direct scan-out for some other element
        let is_plane_compatible = |plane: &&PlaneInfo| {
            previous_frame_state
                .plane_state(plane.handle)
                .map(|state| {
                    state
                        .element_state
                        .as_ref()
                        .map(|state| state.z_index <= element_config.z_index)
                        .unwrap_or(false)
                        && state
                            .config
                            .as_ref()
                            .map(|config| config.properties.is_compatible(&element_config.properties))
                            .unwrap_or(false)
                })
                .unwrap_or(false)
        };

        let mut test_overlay_plane = |plane: &PlaneInfo,
                                      element_config: &ElementPlaneConfig<
            '_,
            <A as Allocator>::Buffer,
            <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer,
        >| {
            // something is already assigned to our overlay plane
            if frame_state.is_assigned(plane.handle) {
                trace!(
                    "skipping {:?} with zpos {:?} for element {:?}, already has element assigned, skipping",
                    plane.handle, plane.zpos, element_id,
                );
                return Err(None);
            }

            // test if the plane represents an underlay
            let is_underlay =
                self.surface.plane_info().zpos.unwrap_or_default() > plane.zpos.unwrap_or_default();

            if is_underlay && !(element_is_opaque && primary_plane_has_alpha) {
                trace!(
                    "skipping direct scan-out on underlay {:?} with zpos {:?}, element {:?} is not opaque or primary plane has no alpha channel",
                    plane.handle, plane.zpos, element_id
                );
                return Err(None);
            }

            // if the element overlaps with an element on
            // the primary plane and is not an underlay
            // we can not assign it to any overlay plane
            if overlaps_with_primary_plane_element && !is_underlay {
                trace!(
                    "skipping direct scan-out on {:?} with zpos {:?}, element {:?} overlaps with element on primary plane",
                    plane.handle, plane.zpos, element_id,
                );
                return Err(None);
            }

            let overlaps_with_plane_underneath = self
                .planes
                .overlay
                .iter()
                .filter(|info| {
                    info.handle != plane.handle
                        && info.zpos.unwrap_or_default() <= plane.zpos.unwrap_or_default()
                })
                .any(|overlapping_plane| {
                    frame_state.overlaps(overlapping_plane.handle, element_config.geometry)
                });

            // if we overlap we a plane below which already
            // has an element assigned we can not use the
            // plane for direct scan-out
            if overlaps_with_plane_underneath {
                trace!(
                    "skipping direct scan-out on {:?} with zpos {:?}, element {:?} geometry {:?} overlaps with plane underneath",
                    plane.handle, plane.zpos, element_id, element_config.geometry,
                );
                return Err(None);
            }

            self.try_assign_plane(element, element_config, plane, scale, frame_state)
        };

        // First try to assign the element to a compatible plane, this can save us
        // from some atomic testing
        for plane in self.planes.overlay.iter().filter(is_plane_compatible) {
            if let Ok(plane_assignment) = test_overlay_plane(plane, &element_config) {
                trace!(
                    "assigned element {:?} geometry {:?} to compatible {:?} with zpos {:?}",
                    element_id, element_config.geometry, plane.handle, plane.zpos,
                );
                return Ok(plane_assignment);
            }
        }

        // If we found no compatible plane fall back to walk all available planes
        let mut rendering_reason: Option<RenderingReason> = None;
        for (index, plane) in self.planes.overlay.iter().enumerate() {
            // if the tested element state already tells us that this failed skip the test
            if element_config.failed_planes.overlay_bitmask & (1 << index) != 0 {
                trace!(
                    "skipping direct scan-out on {:?} with zpos {:?}, element {:?} geometry {:?}, test already known to fail",
                    plane.handle, plane.zpos, element_id, element_config.geometry,
                );
                rendering_reason = rendering_reason.or(Some(RenderingReason::ScanoutFailed));
                continue;
            }

            match test_overlay_plane(plane, &element_config) {
                Ok(plane) => return Ok(plane),
                Err(err) => {
                    // if the test failed save that in the tested element state
                    if let Some(RenderingReason::ScanoutFailed) = err {
                        element_config.failed_planes.overlay_bitmask |= 1 << index;
                    }

                    rendering_reason = rendering_reason.or(err)
                }
            }
        }

        Err(rendering_reason)
    }

    #[allow(clippy::too_many_arguments)]
    #[instrument(level = "trace", skip_all)]
    #[profiling::function]
    fn try_assign_plane<R, E>(
        &self,
        element: &E,
        element_config: &ElementPlaneConfig<
            '_,
            <A as Allocator>::Buffer,
            <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer,
        >,
        plane: &PlaneInfo,
        scale: Scale<f64>,
        frame_state: &mut CompositorFrameState<A, F>,
    ) -> Result<PlaneAssignment, Option<RenderingReason>>
    where
        R: Renderer,
        E: RenderElement<R>,
    {
        let element_id = element.id();

        let plane_claim = match self.surface.claim_plane(plane.handle) {
            Some(claim) => claim,
            None => {
                trace!("failed to claim {:?} for element {:?}", plane.handle, element_id);
                return Err(None);
            }
        };

        // Try to assign the element to a plane
        trace!(
            "testing direct scan-out for element {:?} on {:?} with zpos {:?}: fb: {:?}, element_geometry: {:?}",
            element_id, plane.handle, plane.zpos, &element_config.buffer.fb, element_config.geometry
        );

        if !plane.formats.contains(&element_config.properties.format) {
            trace!(
                "skipping direct scan-out on {:?} with zpos {:?} for element {:?}, format {:?} not supported",
                plane.handle, plane.zpos, element_id, element_config.properties.format,
            );
            return Err(Some(RenderingReason::FormatUnsupported));
        }

        let previous_state = self
            .pending_frame
            .as_ref()
            .map(|pending| &pending.frame)
            .unwrap_or(&self.current_frame);

        let previous_commit = previous_state.plane_state(plane.handle).and_then(|state| {
            state.element_state.as_ref().and_then(|state| {
                if state.id == *element_id {
                    Some(state.commit)
                } else {
                    None
                }
            })
        });

        let element_damage = element.damage_since(scale, previous_commit);
        let has_element_damage = !element_damage.is_empty();

        // Damage were applied buffer transform to be in physical-space. We need to invert it to go
        // back to buffer-coordinate. We'll apply the same transform to the element geometry for
        // scale computation as it's already in physical space.
        let transform = element.transform().invert();
        let damage_clips = if has_element_damage {
            PlaneDamageClips::from_damage(
                self.surface.device_fd(),
                element_config.properties.src,
                element_config.geometry,
                transform,
                transform,
                element_damage,
            )
            .ok()
            .flatten()
        } else {
            None
        };

        let config = PlaneConfig {
            properties: element_config.properties,
            buffer: element_config.buffer.clone(),
            damage_clips,
            plane_claim,
            sync: element_config
                .buffer
                .buffer
                .acquire_point(self.signaled_fence.as_ref()),
        };

        let is_compatible = previous_state
            .plane_state(plane.handle)
            .map(|state| {
                state
                    .config
                    .as_ref()
                    .map(|c| c.is_compatible(&config))
                    .unwrap_or(false)
            })
            .unwrap_or(false);

        // We can only skip the plane update if we have no damage and if
        // the src/dst/alpha properties are unchanged. Also we can not skip if
        // the fb did change (this includes the case where we previously
        // had not assigned anything to the plane)
        let skip = !has_element_damage
            && previous_state
                .plane_state(plane.handle)
                .map(|state| {
                    state
                        .config
                        .as_ref()
                        .map(|c| is_compatible && c.buffer.fb == config.buffer.fb)
                        .unwrap_or(false)
                })
                .unwrap_or(false);

        let plane_state = PlaneState {
            skip,
            needs_test: true,
            element_state: Some(PlaneElementState {
                id: element_id.clone(),
                commit: element.current_commit(),
                z_index: element_config.z_index,
                cursor_size: None,
            }),
            config: Some(config),
        };

        let res = if is_compatible {
            trace!(
                "skipping atomic test for compatible element {:?} on {:?} with zpos {:?}",
                element_id, plane.handle, plane.zpos,
            );
            frame_state.set_state(plane.handle, plane_state);
            true
        } else {
            frame_state
                .test_state(
                    &self.surface,
                    self.supports_fencing,
                    plane.handle,
                    plane_state,
                    false,
                )
                .is_ok()
        };

        if res {
            trace!(
                "successfully assigned element {:?} to {:?} with zpos {:?} for direct scan-out",
                element_id, plane.handle, plane.zpos,
            );

            Ok(plane.into())
        } else {
            trace!(
                "skipping direct scan-out on {:?} with zpos {:?} for element {:?}, test failed",
                plane.handle, plane.zpos, element_id
            );

            Err(Some(RenderingReason::ScanoutFailed))
        }
    }

    /// Clear the surface, setting DPMS state to off, disabling all planes,
    /// and clearing the pending frame.
    ///
    /// Calling [`queue_frame`][Self::queue_frame] will re-enable.
    pub fn clear(&mut self) -> Result<(), DrmError> {
        self.surface.clear()?;

        self.current_frame
            .planes
            .iter_mut()
            .for_each(|(_, state)| *state = Default::default());
        self.pending_frame = None;
        self.queued_frame = None;
        self.next_frame = None;

        Ok(())
    }
}

#[inline]
fn apply_underlying_storage_transform(
    element_transform: Transform,
    storage: &UnderlyingStorage<'_>,
) -> Transform {
    match storage {
        UnderlyingStorage::Wayland(buffer) => {
            if buffer_y_inverted(buffer).unwrap_or(false) {
                match element_transform {
                    Transform::Normal => Transform::Flipped,
                    Transform::_90 => Transform::Flipped90,
                    Transform::_180 => Transform::Flipped180,
                    Transform::_270 => Transform::Flipped270,
                    Transform::Flipped => Transform::Normal,
                    Transform::Flipped90 => Transform::_90,
                    Transform::Flipped180 => Transform::_180,
                    Transform::Flipped270 => Transform::_270,
                }
            } else {
                element_transform
            }
        }
        UnderlyingStorage::Memory { .. } => element_transform,
    }
}

#[inline]
fn apply_output_transform(transform: Transform, output_transform: Transform) -> Transform {
    match (transform, output_transform) {
        (Transform::Normal, output_transform) => output_transform,

        (Transform::_90, Transform::Normal) => Transform::_270,
        (Transform::_90, Transform::_90) => Transform::Normal,
        (Transform::_90, Transform::_180) => Transform::_90,
        (Transform::_90, Transform::_270) => Transform::_180,
        (Transform::_90, Transform::Flipped) => Transform::Flipped270,
        (Transform::_90, Transform::Flipped90) => Transform::Flipped,
        (Transform::_90, Transform::Flipped180) => Transform::Flipped90,
        (Transform::_90, Transform::Flipped270) => Transform::Flipped180,

        (Transform::_180, Transform::Normal) => Transform::_180,
        (Transform::_180, Transform::_90) => Transform::_270,
        (Transform::_180, Transform::_180) => Transform::Normal,
        (Transform::_180, Transform::_270) => Transform::_90,
        (Transform::_180, Transform::Flipped) => Transform::Flipped180,
        (Transform::_180, Transform::Flipped90) => Transform::Flipped270,
        (Transform::_180, Transform::Flipped180) => Transform::Flipped,
        (Transform::_180, Transform::Flipped270) => Transform::Flipped90,

        (Transform::_270, Transform::Normal) => Transform::_90,
        (Transform::_270, Transform::_90) => Transform::_180,
        (Transform::_270, Transform::_180) => Transform::_270,
        (Transform::_270, Transform::_270) => Transform::Normal,
        (Transform::_270, Transform::Flipped) => Transform::Flipped90,
        (Transform::_270, Transform::Flipped90) => Transform::Flipped180,
        (Transform::_270, Transform::Flipped180) => Transform::Flipped270,
        (Transform::_270, Transform::Flipped270) => Transform::Flipped,

        (Transform::Flipped, Transform::Normal) => Transform::Flipped,
        (Transform::Flipped, Transform::_90) => Transform::Flipped90,
        (Transform::Flipped, Transform::_180) => Transform::Flipped180,
        (Transform::Flipped, Transform::_270) => Transform::Flipped270,
        (Transform::Flipped, Transform::Flipped) => Transform::Normal,
        (Transform::Flipped, Transform::Flipped90) => Transform::_90,
        (Transform::Flipped, Transform::Flipped180) => Transform::_180,
        (Transform::Flipped, Transform::Flipped270) => Transform::_270,

        (Transform::Flipped90, Transform::Normal) => Transform::Flipped270,
        (Transform::Flipped90, Transform::_90) => Transform::Flipped,
        (Transform::Flipped90, Transform::_180) => Transform::Flipped90,
        (Transform::Flipped90, Transform::_270) => Transform::Flipped180,
        (Transform::Flipped90, Transform::Flipped) => Transform::_270,
        (Transform::Flipped90, Transform::Flipped90) => Transform::Normal,
        (Transform::Flipped90, Transform::Flipped180) => Transform::_90,
        (Transform::Flipped90, Transform::Flipped270) => Transform::_180,

        (Transform::Flipped180, Transform::Normal) => Transform::Flipped180,
        (Transform::Flipped180, Transform::_90) => Transform::Flipped270,
        (Transform::Flipped180, Transform::_180) => Transform::Flipped,
        (Transform::Flipped180, Transform::_270) => Transform::Flipped90,
        (Transform::Flipped180, Transform::Flipped) => Transform::_180,
        (Transform::Flipped180, Transform::Flipped90) => Transform::_270,
        (Transform::Flipped180, Transform::Flipped180) => Transform::Normal,
        (Transform::Flipped180, Transform::Flipped270) => Transform::_90,

        (Transform::Flipped270, Transform::Normal) => Transform::Flipped90,
        (Transform::Flipped270, Transform::_90) => Transform::Flipped180,
        (Transform::Flipped270, Transform::_180) => Transform::Flipped270,
        (Transform::Flipped270, Transform::_270) => Transform::Flipped,
        (Transform::Flipped270, Transform::Flipped) => Transform::_90,
        (Transform::Flipped270, Transform::Flipped90) => Transform::_180,
        (Transform::Flipped270, Transform::Flipped180) => Transform::_270,
        (Transform::Flipped270, Transform::Flipped270) => Transform::Normal,
    }
}

#[profiling::function]
fn copy_element_to_cursor_bo<R, E>(
    renderer: &mut R,
    element: &E,
    element_size: Size<i32, Physical>,
    cursor_size: Size<i32, Physical>,
    output_transform: Transform,
    bo: &mut GbmBuffer,
) -> bool
where
    R: Renderer,
    E: RenderElement<R>,
{
    // Without access to the underlying storage we can not copy anything
    let Some(underlying_storage) = element.underlying_storage(renderer) else {
        return false;
    };

    let element_src = element.src();
    let element_scale = element_src.size / element_size.to_f64();

    // We only copy if no crop, scale or transform is active
    if element_src.loc != Point::default()
        || element_scale != Scale::from(1f64)
        || element.transform() != Transform::Normal
        || output_transform != Transform::Normal
    {
        return false;
    }

    let bo_format = bo.format().code;
    let bo_stride = bo.stride();

    let mut copy_to_bo = |src, src_stride, src_height| {
        if src_stride == bo_stride as i32 {
            bo.write(src).is_ok()
        } else {
            let res = bo.map_mut(0, 0, cursor_size.w as u32, cursor_size.h as u32, |mbo| {
                let dst = mbo.buffer_mut();
                for row in 0..src_height {
                    let src_row_start = (row * src_stride) as usize;
                    let src_row_end = src_row_start + src_stride as usize;
                    let src_row = &src[src_row_start..src_row_end];
                    let dst_row_start = (row * bo_stride as i32) as usize;
                    let dst_row_end = dst_row_start + src_stride as usize;
                    let dst_row = &mut dst[dst_row_start..dst_row_end];
                    dst_row.copy_from_slice(src_row);
                }
            });
            res.is_ok()
        }
    };

    match underlying_storage {
        UnderlyingStorage::Wayland(buffer) => {
            // Only shm buffers are supported for copy
            shm::with_buffer_contents(buffer, |ptr, len, data| {
                let Some(format) = shm::shm_format_to_fourcc(data.format) else {
                    return false;
                };

                if format != bo_format {
                    return false;
                };

                let expected_len = (data.stride * data.height) as usize;
                if data.offset as usize + expected_len > len {
                    return false;
                };

                copy_to_bo(
                    unsafe { std::slice::from_raw_parts(ptr.offset(data.offset as isize), expected_len) },
                    data.stride,
                    data.height,
                )
            })
            .unwrap_or(false)
        }
        UnderlyingStorage::Memory(memory) => {
            if memory.format() != bo_format {
                return false;
            };

            copy_to_bo(memory, memory.stride(), memory.size().h)
        }
    }
}

struct CachedDrmFramebuffer<B: Framebuffer>(Arc<DrmFramebuffer<B>>);

impl<B: Framebuffer> PartialEq for CachedDrmFramebuffer<B> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        AsRef::<framebuffer::Handle>::as_ref(&self) == AsRef::<framebuffer::Handle>::as_ref(&other)
    }
}

impl<B: Framebuffer + std::fmt::Debug> std::fmt::Debug for CachedDrmFramebuffer<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("CachedDrmFramebuffer").field(&self.0).finish()
    }
}

impl<B: Framebuffer> CachedDrmFramebuffer<B> {
    #[inline]
    fn new(buffer: DrmFramebuffer<B>) -> Self {
        CachedDrmFramebuffer(Arc::new(buffer))
    }
}

impl<B: Framebuffer> Clone for CachedDrmFramebuffer<B> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<B: Framebuffer> AsRef<framebuffer::Handle> for CachedDrmFramebuffer<B> {
    #[inline]
    fn as_ref(&self) -> &framebuffer::Handle {
        (*self.0).as_ref()
    }
}

impl<B: Framebuffer> Framebuffer for CachedDrmFramebuffer<B> {
    #[inline]
    fn format(&self) -> drm_fourcc::DrmFormat {
        (*self.0).format()
    }
}

/// Errors thrown by a [`DrmCompositor`]
#[derive(Debug, thiserror::Error)]
pub enum FrameError<
    A: std::error::Error + Send + Sync + 'static,
    B: std::error::Error + Send + Sync + 'static,
    F: std::error::Error + Send + Sync + 'static,
> {
    /// Failed to claim the primary plane
    #[error("Failed to claim the primary plane")]
    PrimaryPlaneClaimFailed,
    /// No supported pixel format for the given plane could be determined
    #[error("No supported plane buffer format found")]
    NoSupportedPlaneFormat,
    /// No supported pixel format for the given renderer could be determined
    #[error("No supported renderer buffer format found")]
    NoSupportedRendererFormat,
    /// The swapchain is exhausted, you need to call `frame_submitted`
    #[error("Failed to allocate a new buffer")]
    NoFreeSlotsError,
    /// Error accessing the drm device
    #[error("The underlying drm surface encountered an error: {0}")]
    DrmError(#[from] DrmError),
    /// Error during buffer allocation
    #[error("The underlying allocator encountered an error: {0}")]
    Allocator(#[source] A),
    /// Error during exporting the buffer as dmabuf
    #[error("Failed to export the allocated buffer as dmabuf: {0}")]
    AsDmabufError(#[source] B),
    /// Error during exporting a framebuffer
    #[error("The framebuffer export encountered an error: {0}")]
    FramebufferExport(#[source] F),
    /// No framebuffer available
    #[error("No framebuffer available")]
    NoFramebuffer,
    /// The frame is empty
    ///
    /// Possible reasons include not calling `render_frame` prior to
    /// `queue_frame` or trying to queue a frame without changes.
    #[error("No frame has been prepared or it does not contain any changes")]
    EmptyFrame,
}

/// Error returned from [`DrmCompositor::render_frame`]
#[derive(thiserror::Error)]
pub enum RenderFrameError<
    A: std::error::Error + Send + Sync + 'static,
    B: std::error::Error + Send + Sync + 'static,
    F: std::error::Error + Send + Sync + 'static,
    R: std::error::Error,
> {
    /// Preparing the frame encountered an error
    #[error(transparent)]
    PrepareFrame(#[from] FrameError<A, B, F>),
    /// Rendering the frame encountered en error
    #[error(transparent)]
    RenderFrame(#[from] OutputDamageTrackerError<R>),
}

impl<A, B, F, R> std::fmt::Debug for RenderFrameError<A, B, F, R>
where
    A: std::error::Error + Send + Sync + 'static,
    B: std::error::Error + Send + Sync + 'static,
    F: std::error::Error + Send + Sync + 'static,
    R: std::error::Error,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PrepareFrame(arg0) => f.debug_tuple("PrepareFrame").field(arg0).finish(),
            Self::RenderFrame(arg0) => f.debug_tuple("RenderFrame").field(arg0).finish(),
        }
    }
}

impl<
    A: std::error::Error + Send + Sync + 'static,
    B: std::error::Error + Send + Sync + 'static,
    F: std::error::Error + Send + Sync + 'static,
> From<FrameError<A, B, F>> for SwapBuffersError
{
    #[inline]
    fn from(err: FrameError<A, B, F>) -> SwapBuffersError {
        match err {
            x @ FrameError::NoSupportedPlaneFormat
            | x @ FrameError::NoSupportedRendererFormat
            | x @ FrameError::PrimaryPlaneClaimFailed
            | x @ FrameError::NoFramebuffer => SwapBuffersError::ContextLost(Box::new(x)),
            x @ FrameError::NoFreeSlotsError | x @ FrameError::EmptyFrame => {
                SwapBuffersError::TemporaryFailure(Box::new(x))
            }
            FrameError::DrmError(err) => err.into(),
            FrameError::Allocator(err) => SwapBuffersError::ContextLost(Box::new(err)),
            FrameError::AsDmabufError(err) => SwapBuffersError::ContextLost(Box::new(err)),
            FrameError::FramebufferExport(err) => SwapBuffersError::ContextLost(Box::new(err)),
        }
    }
}

fn nvidia_drm_version() -> Option<(u32, u32, u32)> {
    let ver = std::fs::read_to_string("/sys/module/nvidia_drm/version").ok()?;
    let mut components = ver.trim().split('.');
    let major = u32::from_str(components.next()?).ok()?;
    let minor = u32::from_str(components.next()?).ok()?;
    let patch = u32::from_str(components.next()?).ok()?;
    Some((major, minor, patch))
}

#[test]
fn drm_compositor_is_send() {
    use std::marker::PhantomData;

    use crate::backend::drm::DrmDeviceFd;

    fn is_send<T: Send>() {
        let _ = PhantomData::<T>;
    }

    is_send::<DrmCompositor<GbmAllocator<DrmDeviceFd>, GbmFramebufferExporter<DrmDeviceFd>, (), DrmDeviceFd>>(
    );
}

#[cfg(test)]
mod async_flip_tests {
    use std::cell::RefCell;

    use drm::control::{RawResourceHandle, from_u32, plane};
    use drm_fourcc::{DrmFormat, DrmFourcc, DrmModifier};
    use rustix::io::Errno;

    use super::{
        AsyncFlipBlocker, AsyncFlipPlane, AsyncFlipSurface, FlipOutcome, PlaneProperties, async_flip_blocker,
        flip_frame,
    };
    use crate::{
        backend::{
            drm::{DrmError, error::AccessError},
            renderer::PresentationMode,
        },
        utils::{Rectangle, Transform},
    };

    fn handle<T: From<RawResourceHandle>>(raw: u32) -> T {
        from_u32(raw).unwrap()
    }

    const PRIMARY: u32 = 1;
    const CURSOR: u32 = 2;
    const OVERLAY: u32 = 3;

    fn properties() -> PlaneProperties {
        PlaneProperties {
            src: Rectangle::from_size((2560.0, 1440.0).into()),
            dst: Rectangle::from_size((2560, 1440).into()),
            transform: Transform::Normal,
            alpha: 1.0,
            format: DrmFormat {
                code: DrmFourcc::Xrgb2101010,
                modifier: DrmModifier::Linear,
            },
        }
    }

    /// A client buffer scanned out directly
    fn client(raw_fb: u32) -> AsyncFlipPlane {
        AsyncFlipPlane {
            config: Some((handle(raw_fb), properties())),
            skip: false,
            client_buffer: true,
        }
    }

    fn unused() -> AsyncFlipPlane {
        AsyncFlipPlane {
            config: None,
            skip: true,
            client_buffer: false,
        }
    }

    fn cursor(x: i32, skip: bool) -> AsyncFlipPlane {
        AsyncFlipPlane {
            config: Some((
                handle(100),
                PlaneProperties {
                    src: Rectangle::from_size((64.0, 64.0).into()),
                    dst: Rectangle::new((x, 10).into(), (64, 64).into()),
                    ..properties()
                },
            )),
            skip,
            client_buffer: false,
        }
    }

    /// A surface that allows async page flips
    fn surface() -> AsyncFlipSurface {
        AsyncFlipSurface {
            supported: true,
            full_commit: false,
            vrr_commit_pending: false,
            composited: false,
        }
    }

    type Planes = Vec<(plane::Handle, Option<AsyncFlipPlane>, AsyncFlipPlane)>;

    /// The planes of a frame with this primary plane (on screen, next), an unused overlay plane
    /// and these other planes
    fn planes(
        primary: (Option<AsyncFlipPlane>, AsyncFlipPlane),
        others: impl IntoIterator<Item = (u32, Option<AsyncFlipPlane>, AsyncFlipPlane)>,
    ) -> Planes {
        let mut planes: Planes = vec![(handle(OVERLAY), Some(unused()), unused())];
        planes.extend(
            others
                .into_iter()
                .map(|(raw, current, next)| (handle(raw), current, next)),
        );
        // The primary plane is not the first one
        planes.push((handle(PRIMARY), primary.0, primary.1));
        planes
    }

    fn blocker(surface: AsyncFlipSurface, planes: Planes) -> Option<AsyncFlipBlocker> {
        async_flip_blocker(surface, handle(PRIMARY), planes)
    }

    #[test]
    fn a_new_buffer_on_an_unchanged_primary_plane_flips_async() {
        let primary = (Some(client(1)), client(2));
        assert_eq!(blocker(surface(), planes(primary, [])), None);
        // A cursor that is in use but neither moves nor changes
        let still_cursor = (CURSOR, Some(cursor(10, true)), cursor(10, true));
        assert_eq!(blocker(surface(), planes(primary, [still_cursor])), None);
        // What was on screen before does not have to be a client buffer
        let composited = AsyncFlipPlane {
            client_buffer: false,
            ..client(1)
        };
        assert_eq!(
            blocker(surface(), planes((Some(composited), client(2)), [])),
            None
        );
    }

    #[test]
    fn the_surface_has_to_allow_async_flips() {
        let primary = (Some(client(1)), client(2));
        let cases = [
            (
                AsyncFlipSurface {
                    supported: false,
                    ..surface()
                },
                AsyncFlipBlocker::Unsupported,
            ),
            (
                AsyncFlipSurface {
                    full_commit: true,
                    ..surface()
                },
                AsyncFlipBlocker::FullCommit,
            ),
            // `use_vrr` switched VRR without a modeset: the next flip has to send `VRR_ENABLED`
            (
                AsyncFlipSurface {
                    vrr_commit_pending: true,
                    ..surface()
                },
                AsyncFlipBlocker::VrrChange,
            ),
        ];
        for (surface, expected) in cases {
            assert_eq!(
                blocker(surface, planes(primary, [])),
                Some(expected),
                "{surface:?}"
            );
        }
    }

    #[test]
    fn a_composited_frame_flips_async_only_when_allowed() {
        let swapchain = |raw_fb: u32| AsyncFlipPlane {
            client_buffer: false,
            ..client(raw_fb)
        };
        let composited = AsyncFlipSurface {
            composited: true,
            ..surface()
        };
        let primary = (Some(swapchain(1)), swapchain(2));
        assert_eq!(
            blocker(surface(), planes(primary, [])),
            Some(AsyncFlipBlocker::NoDirectScanout)
        );
        assert_eq!(blocker(composited, planes(primary, [])), None);
        // A client buffer still flips async with the flag
        assert_eq!(
            blocker(composited, planes((Some(client(1)), client(2)), [])),
            None
        );
        // Every other rule still holds: an unchanged cursor is fine, a moved one is not
        let still_cursor = (CURSOR, Some(cursor(10, true)), cursor(10, true));
        assert_eq!(blocker(composited, planes(primary, [still_cursor])), None);
        let moved_cursor = (CURSOR, Some(cursor(10, true)), cursor(11, false));
        assert_eq!(
            blocker(composited, planes(primary, [moved_cursor])),
            Some(AsyncFlipBlocker::OtherPlaneChanged)
        );
        // The cursor plane switched off (the cursor is drawn into the frame from now on)
        let cursor_off = (CURSOR, Some(cursor(10, true)), unused());
        assert_eq!(
            blocker(composited, planes(primary, [cursor_off])),
            Some(AsyncFlipBlocker::OtherPlaneChanged)
        );
        // A primary plane that is off never flips async
        assert_eq!(
            blocker(composited, planes((Some(swapchain(1)), unused()), [])),
            Some(AsyncFlipBlocker::NoDirectScanout)
        );
        // Neither does a full commit or a pending VRR switch
        for surface in [
            AsyncFlipSurface {
                full_commit: true,
                ..composited
            },
            AsyncFlipSurface {
                vrr_commit_pending: true,
                ..composited
            },
        ] {
            assert!(blocker(surface, planes(primary, [])).is_some(), "{surface:?}");
        }
    }

    #[test]
    fn the_primary_plane_must_scan_out_a_client_buffer() {
        let composited = AsyncFlipPlane {
            client_buffer: false,
            ..client(2)
        };
        for next in [composited, unused()] {
            assert_eq!(
                blocker(surface(), planes((Some(client(1)), next), [])),
                Some(AsyncFlipBlocker::NoDirectScanout),
                "{next:?}"
            );
        }
        // No primary plane in the frame at all
        let without_primary = vec![(handle(OVERLAY), Some(unused()), unused())];
        assert_eq!(
            blocker(surface(), without_primary),
            Some(AsyncFlipBlocker::NoDirectScanout)
        );
    }

    #[test]
    fn the_primary_plane_may_change_nothing_but_the_framebuffer() {
        let with = |properties: PlaneProperties| AsyncFlipPlane {
            config: Some((handle(2), properties)),
            ..client(2)
        };
        let changed = [
            with(PlaneProperties {
                src: Rectangle::from_size((1920.0, 1080.0).into()),
                ..properties()
            }),
            with(PlaneProperties {
                dst: Rectangle::new((0, 1).into(), (2560, 1440).into()),
                ..properties()
            }),
            with(PlaneProperties {
                transform: Transform::_180,
                ..properties()
            }),
            with(PlaneProperties {
                alpha: 0.5,
                ..properties()
            }),
            with(PlaneProperties {
                format: DrmFormat {
                    code: DrmFourcc::Xrgb8888,
                    modifier: DrmModifier::Linear,
                },
                ..properties()
            }),
            with(PlaneProperties {
                format: DrmFormat {
                    code: DrmFourcc::Xrgb2101010,
                    modifier: DrmModifier::from(0x0200_0000_0000_0001u64),
                },
                ..properties()
            }),
        ];
        for next in changed {
            assert_eq!(
                blocker(surface(), planes((Some(client(1)), next), [])),
                Some(AsyncFlipBlocker::PrimaryChanged),
                "{next:?}"
            );
        }
        // Nothing on screen yet
        for current in [Some(unused()), None] {
            assert_eq!(
                blocker(surface(), planes((current, client(2)), [])),
                Some(AsyncFlipBlocker::PrimaryChanged)
            );
        }
    }

    #[test]
    fn no_other_plane_may_change() {
        let changes = [
            // A new cursor image
            (Some(cursor(10, true)), cursor(10, false)),
            // A cursor move skipped by `SKIP_CURSOR_ONLY_UPDATES` still changes the plane
            (Some(cursor(10, true)), cursor(11, true)),
            // A plane is turned on, or off
            (Some(unused()), cursor(10, false)),
            (
                Some(cursor(10, true)),
                AsyncFlipPlane {
                    skip: false,
                    ..unused()
                },
            ),
        ];
        for (current, next) in changes {
            assert_eq!(
                blocker(
                    surface(),
                    planes((Some(client(1)), client(2)), [(CURSOR, current, next)])
                ),
                Some(AsyncFlipBlocker::OtherPlaneChanged),
                "{next:?}"
            );
        }
    }

    fn error(errno: Errno) -> DrmError {
        DrmError::Access(AccessError {
            errmsg: "Page flip commit failed",
            dev: None,
            source: errno.into(),
        })
    }

    fn errno(result: &Result<(), DrmError>) -> Option<Errno> {
        match result {
            Err(DrmError::Access(access)) => Errno::from_io_error(&access.source),
            _ => None,
        }
    }

    /// Runs `flip_frame` against a kernel that answers the async commit with `async_result` and
    /// the vsynced one with `vsync_result`, and returns the outcome and the commits made
    fn flip(
        try_async: bool,
        async_result: Option<Errno>,
        vsync_result: Option<Errno>,
    ) -> (FlipOutcome, Vec<PresentationMode>) {
        let commits = RefCell::new(Vec::new());
        let outcome = flip_frame(try_async, |mode| {
            commits.borrow_mut().push(mode);
            let result = match mode {
                PresentationMode::Async => async_result,
                PresentationMode::VSync => vsync_result,
            };
            result.map_or(Ok(()), |errno| Err(error(errno)))
        });
        (outcome, commits.into_inner())
    }

    #[test]
    fn a_frame_without_async_flips_vsynced() {
        let (outcome, commits) = flip(false, None, None);
        assert!(outcome.result.is_ok());
        assert_eq!(outcome.presentation_mode, PresentationMode::VSync);
        assert!(!outcome.async_refused);
        assert_eq!(commits, [PresentationMode::VSync]);

        let (outcome, commits) = flip(false, None, Some(Errno::INVAL));
        assert_eq!(errno(&outcome.result), Some(Errno::INVAL));
        assert!(!outcome.async_refused);
        assert_eq!(commits, [PresentationMode::VSync]);
    }

    #[test]
    fn an_accepted_async_flip_is_presented_async() {
        let (outcome, commits) = flip(true, None, None);
        assert!(outcome.result.is_ok());
        assert_eq!(outcome.presentation_mode, PresentationMode::Async);
        assert!(!outcome.async_refused);
        assert_eq!(commits, [PresentationMode::Async]);
    }

    #[test]
    fn a_refused_async_flip_is_flipped_vsynced_at_once() {
        let (outcome, commits) = flip(true, Some(Errno::INVAL), None);
        assert!(outcome.result.is_ok());
        assert_eq!(outcome.presentation_mode, PresentationMode::VSync);
        assert!(outcome.async_refused);
        assert_eq!(commits, [PresentationMode::Async, PresentationMode::VSync]);

        // The error of the vsynced flip is the one returned
        let (outcome, commits) = flip(true, Some(Errno::INVAL), Some(Errno::NOMEM));
        assert_eq!(errno(&outcome.result), Some(Errno::NOMEM));
        assert_eq!(outcome.presentation_mode, PresentationMode::VSync);
        assert!(outcome.async_refused);
        assert_eq!(commits, [PresentationMode::Async, PresentationMode::VSync]);
    }

    #[test]
    fn other_errors_of_the_async_flip_are_returned() {
        // `EBUSY`: a vsynced non-blocking commit would get it as well
        for errno_ in [Errno::BUSY, Errno::ACCESS, Errno::PERM, Errno::NOMEM] {
            let (outcome, commits) = flip(true, Some(errno_), None);
            assert_eq!(errno(&outcome.result), Some(errno_));
            assert_eq!(outcome.presentation_mode, PresentationMode::VSync);
            assert!(!outcome.async_refused);
            assert_eq!(commits, [PresentationMode::Async]);
        }
        let outcome = flip_frame(true, |_| Err(DrmError::DeviceInactive));
        assert!(matches!(outcome.result, Err(DrmError::DeviceInactive)));
        assert!(!outcome.async_refused);
    }
}
