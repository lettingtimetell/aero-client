//! Windows Media Foundation: the only place in the client that talks to it.
//!
//! * `api` loads the Media Foundation entry points at run time and owns the per-thread session.
//! * [`probe`] lists the encoders this PC has.
//! * [`writer`] is the MP4 sink writer.
//! * [`verify`] decodes a finished file with Windows' own decoders (tests, checks).

pub(crate) mod api;
// The hardware encoder on the render adapter.
pub(crate) mod device;
pub mod probe;
pub mod verify;
pub mod writer;
