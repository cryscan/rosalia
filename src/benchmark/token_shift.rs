//! Benchmark for the `token_shift` compute shader.
//!
//! # Token Shift Operation
//!
//! The token shift operation implements time-mix interpolation, a core component of
//! RWKV-style architectures. For each channel of each token, it computes a weighted
//! blend between the current value and the previous value (from the prior token or
//! the initial state), where the blend weight is given by a per-channel time-mix factor.
//!
//! ## Dimensions
//!
//! - **Channel (C)**: The embedding dimension of a single layer. Each channel has its
//!   own independent time-mix factor, enabling fine-grained interpolation control.
//! - **Token (T)**: The number of input tokens in the current batch. Tokens are grouped
//!   into independent sequences (states) via the cursor mechanism.
//! - **State**: In continuous batching, the token dimension is divided into multiple
//!   independent sequences of varying lengths. Each such sequence is called a "state".
//!   The state ID determines which initial state vector to use for the first token
//!   of that sequence.
//! - **Batch**: Allows processing multiple independent groups (e.g., q/k/v vectors)
//!   in a single dispatch. Each batch has its own set of states and tokens.
//!
//! ## Cursor Mechanism
//!
//! Each token position is associated with a packed 32-bit cursor value that encodes:
//! - `state` (bits 0–7): Which state sequence this token belongs to. This determines
//!   which row of the state tensor (input_s) to read the initial value from.
//! - `token` (bits 8–23): The position of this token within its state sequence.
//!   When `token == 0`, the previous value comes from the state tensor; otherwise
//!   it comes from the previous token in the input tensor.
//! - `len` (bits 24–31): The total length of this state sequence (currently unused
//!   in the shader but reserved for future use).
//!
//! ## Normal vs Reversed Mode
//!
//! - **Normal mode** (no `REVERSED` define): `result = mix(prev, value, factor)`
//!   — Interpolates from previous to current, with `factor` controlling how much
//!   of the current value to use. This is the standard RWKV time-mixing behavior.
//! - **Reversed mode** (`REVERSED` define): `result = mix(value, prev, factor)`
//!   — Interpolates from current to previous, with `factor` controlling how much
//!   of the previous value to use. Used for the reciprocal direction in dual-direction
//!   time mixing.

use std::error::Error;

use half::f16;
use vulkanalia::prelude::v1_4::*;

use super::{Bench, create_data};
use crate::{
    app::{App, CommandBuffer, Tensor, Uniform},
    asset::Asset,
    layout::{IndexFn, Layout},
    num::{DataType, FromI8, Scalar},
};

#[cfg(feature = "correctness")]
const REPEAT: usize = 1;
#[cfg(not(feature = "correctness"))]
const REPEAT: usize = 10;
#[cfg(not(feature = "correctness"))]
const WARMUP: usize = 5;

/// Packs a cursor as: state (8 bits) | token (16 bits) | len (8 bits).
fn pack_cursor(state: u32, token: u32, len: u32) -> u32 {
    (state & 0xff) | ((token & 0xffff) << 8) | ((len & 0xff) << 24)
}

/// Benchmark for the token_shift compute shader.
///
/// Validates the correctness of token_shift.comp for time-mix interpolation.
/// The shader computes:
/// - Normal mode: `result = mix(prev, value, factor)` where `prev` is the previous
///   token's value (or the state value for the first token in a sequence).
/// - Reversed mode: `result = mix(value, prev, factor)`.
#[derive(Debug)]
pub struct TokenShiftBench<I, O> {
    pub app: App,

    pub channel: usize,
    pub num_tokens: usize,
    pub num_states: usize,
    pub batch: usize,
    pub reversed: bool,

    pub layout_x: Layout,
    pub layout_s: Layout,
    pub layout_t: Layout,
    pub layout_y: Layout,

    pub data_x: Vec<I>,
    pub data_s: Vec<I>,
    pub data_t: Vec<I>,

    pub cursors: Vec<u32>,

    pub tensor_x_host: Tensor<I>,
    pub tensor_s_host: Tensor<I>,
    pub tensor_t_host: Tensor<I>,
    pub tensor_cursors_host: Tensor<u32>,
    pub tensor_y_host: Tensor<O>,

    pub tensor_x_device: Tensor<I>,
    pub tensor_s_device: Tensor<I>,
    pub tensor_t_device: Tensor<I>,
    pub tensor_cursors_device: Tensor<u32>,
    pub tensor_y_device: Tensor<O>,

    pub params: Uniform,

    pub transfer: CommandBuffer,
    pub compute: CommandBuffer,
}

impl<I, O> TokenShiftBench<I, O>
where
    I: Scalar + FromI8,
    O: Scalar + FromI8,
{
    /// Creates a new token shift benchmark.
    ///
    /// # Arguments
    ///
    /// - `app`: The Vulkan application instance
    /// - `channel`: The embedding dimension (C)
    /// - `num_tokens`: The total number of tokens (T)
    /// - `num_states`: The number of independent state sequences
    /// - `batch`: The batch size
    /// - `reversed`: Whether to use the REVERSED shader variant
    pub fn new(
        app: &App,
        channel: usize,
        num_tokens: usize,
        num_states: usize,
        batch: usize,
        reversed: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let app = app.clone();

        // Layouts follow the shader's small-axis-first convention:
        //   X (input):  (channel, num_tokens, batch)
        //   S (state):  (channel, num_states, batch)
        //   T (time_mix): (channel, num_tokens, batch)
        //   Y (output): (channel, num_tokens, batch)
        //   Cursors:    (num_tokens)
        let layout_x = Layout::from_shape([channel, num_tokens, batch]);
        let layout_s = Layout::from_shape([channel, num_states, batch]);
        let layout_t = Layout::from_shape([channel, num_tokens, batch]);
        let layout_y = Layout::from_shape([channel, num_tokens, batch]);

        let data_x = create_data(&layout_x);
        let data_s = create_data(&layout_s);
        let data_t = create_data(&layout_t);

        // Generate cursors that distribute tokens across states.
        // Each cursor encodes: state_id, position_within_state, sequence_length.
        let cursors = generate_cursors(num_tokens, num_states);

        let layout_cursors = Layout::from_shape([num_tokens]);

        // host tensors (mapped)
        let tensor_x_host = app.create_tensor(&layout_x, None, true)?;
        let tensor_s_host = app.create_tensor(&layout_s, None, true)?;
        let tensor_t_host = app.create_tensor(&layout_t, None, true)?;
        let tensor_cursors_host = app.create_tensor(&layout_cursors, None, true)?;
        let tensor_y_host = app.create_tensor(&layout_y, None, true)?;

        tensor_x_host.copy_from(&data_x, 0)?;
        tensor_s_host.copy_from(&data_s, 0)?;
        tensor_t_host.copy_from(&data_t, 0)?;
        tensor_cursors_host.copy_from(&cursors, 0)?;
        tensor_y_host.clear()?;

        // device tensors (not mapped)
        let tensor_x_device = app.create_tensor(&layout_x, None, false)?;
        let tensor_s_device = app.create_tensor(&layout_s, None, false)?;
        let tensor_t_device = app.create_tensor(&layout_t, None, false)?;
        let tensor_cursors_device = app.create_tensor(&layout_cursors, None, false)?;
        let tensor_y_device = app.create_tensor(&layout_y, None, false)?;

        // uniform buffer with device addresses (matching Params struct in shader)
        let params = [
            tensor_x_device.address,
            tensor_s_device.address,
            tensor_t_device.address,
            tensor_cursors_device.address,
            tensor_y_device.address,
        ];
        let params = app
            .create_uniform(size_of_val(&params))?
            .copy_from(&params)?;

        let transfer = app.allocate_transfer_command_buffers(3)?;
        let compute = app.allocate_compute_command_buffers(1)?;

        unsafe {
            // transfer input tensors to device
            let info = vk::CommandBufferBeginInfo::builder();
            app.device.begin_command_buffer(transfer[0], &info)?;
            tensor_x_device.cmd_copy_from(transfer[0], &tensor_x_host);
            tensor_s_device.cmd_copy_from(transfer[0], &tensor_s_host);
            tensor_t_device.cmd_copy_from(transfer[0], &tensor_t_host);
            tensor_cursors_device.cmd_copy_from(transfer[0], &tensor_cursors_host);
            app.device.end_command_buffer(transfer[0])?;

            // transfer output initialization data to device
            let info = vk::CommandBufferBeginInfo::builder();
            app.device.begin_command_buffer(transfer[1], &info)?;
            tensor_y_device.cmd_copy_from(transfer[1], &tensor_y_host);
            app.device.end_command_buffer(transfer[1])?;

            // transfer output from device to host
            let info = vk::CommandBufferBeginInfo::builder();
            app.device.begin_command_buffer(transfer[2], &info)?;
            tensor_y_host.cmd_copy_from(transfer[2], &tensor_y_device);
            app.device.end_command_buffer(transfer[2])?;
        }

        Ok(Self {
            app,
            channel,
            num_tokens,
            num_states,
            batch,
            reversed,
            layout_x,
            layout_s,
            layout_t,
            layout_y,
            data_x,
            data_s,
            data_t,
            cursors,
            tensor_x_host,
            tensor_s_host,
            tensor_t_host,
            tensor_cursors_host,
            tensor_y_host,
            tensor_x_device,
            tensor_s_device,
            tensor_t_device,
            tensor_cursors_device,
            tensor_y_device,
            params,
            transfer,
            compute,
        })
    }
}

impl<I, O> TokenShiftBench<I, O>
where
    I: Scalar,
    O: Scalar,
{
    pub fn matches_type(&self, i: DataType, o: DataType) -> bool {
        (i, o) == (I::DATA_TYPE, O::DATA_TYPE)
    }
}

impl Bench for TokenShiftBench<f16, f16> {
    type Output = f16;

    fn compute_ans(&self) -> Vec<Self::Output> {
        let Self {
            data_x,
            data_s,
            data_t,
            cursors,
            layout_x,
            layout_s,
            layout_t,
            layout_y,
            channel,
            batch,
            reversed,
            ..
        } = self;

        let mut ans = vec![f16::ZERO; layout_y.co_size()];

        for b in 0..*batch {
            for (token, &cursor) in cursors.iter().enumerate() {
                let state_id = cursor & 0xff;
                let token_id = (cursor >> 8) & 0xffff;

                for c in 0..*channel {
                    let x_idx = layout_x.value([c, token, b]);
                    let t_idx = layout_t.value([c, token, b]);
                    let value: f32 = data_x[x_idx].to_f32();
                    let factor: f32 = data_t[t_idx].to_f32();

                    let prev: f32 = if token_id == 0 {
                        // first token in sequence: use state value
                        let s_idx = layout_s.value([c, state_id as usize, b]);
                        data_s[s_idx].to_f32()
                    } else {
                        // subsequent tokens: use previous token's value
                        let prev_idx = layout_x.value([c, token - 1, b]);
                        data_x[prev_idx].to_f32()
                    };

                    let result = if *reversed {
                        mix_f32(value, prev, factor)
                    } else {
                        mix_f32(prev, value, factor)
                    };

                    let y_idx = layout_y.value([c, token, b]);
                    ans[y_idx] = f16::from_f32(result);
                }
            }
        }

        ans
    }

    fn check_ans(&self, actual: &[Self::Output]) -> usize {
        self.layout_y
            .iter_indices()
            .map(|(_, value)| (actual[value].to_f32(), self.tensor_y_host[value].to_f32()))
            .filter(|(x, y)| (x - y).abs() > 1e-2) // tolerance for f16
            .count()
    }
}

/// GLSL-style mix: mix(a, b, t) = a * (1 - t) + b * t
fn mix_f32(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

/// Generates cursors that distribute tokens round-robin across states.
///
/// Each state gets approximately the same number of tokens. The cursor for each
/// token encodes its state ID, its position within that state's sequence, and
/// the total length of that state's sequence.
fn generate_cursors(num_tokens: usize, num_states: usize) -> Vec<u32> {
    let base_len = num_tokens / num_states;
    let remainder = num_tokens % num_states;

    // Compute the length of each state sequence
    let state_lens: Vec<u32> = (0..num_states)
        .map(|s| {
            if s < remainder {
                (base_len + 1) as u32
            } else {
                base_len as u32
            }
        })
        .collect();

    // Build cursor for each token position
    let mut cursors = Vec::with_capacity(num_tokens);
    let mut state_pos = vec![0u32; num_states];
    for t in 0..num_tokens {
        let state = t % num_states;
        let pos = state_pos[state];
        state_pos[state] += 1;
        cursors.push(pack_cursor(state as u32, pos, state_lens[state]));
    }

    cursors
}

impl<I, O> TokenShiftBench<I, O>
where
    I: Scalar,
    O: Scalar,
    TokenShiftBench<I, O>: Bench<Output = O>,
{
    pub fn benchmark_token_shift(&self) -> Result<(), Box<dyn Error>> {
        let Self {
            app,
            layout_x,
            layout_s,
            layout_t,
            layout_y,
            params,
            transfer,
            compute,
            channel,
            num_tokens,
            num_states,
            batch,
            reversed,
            ..
        } = self;

        #[cfg(feature = "correctness")]
        let ans = self.compute_ans();

        unsafe {
            // transfer input tensors to device
            let info = vk::SubmitInfo::builder().command_buffers(&transfer[0..=0]);
            let queue = app.transfer.queue;
            app.device.queue_submit(queue, &[info], vk::Fence::null())?;
            app.device.queue_wait_idle(queue)?;
        }

        // check type match
        if !self.matches_type(DataType::F16, DataType::F16) {
            return Err("Type mismatch: only f16 -> f16 is supported".into());
        }

        // load shader
        let dt_i = I::DATA_TYPE.to_string().to_lowercase();
        let dt_o = O::DATA_TYPE.to_string().to_lowercase();
        let path = if *reversed {
            format!("shaders/spv/token_shift_{dt_i}_{dt_o}_rev.spv")
        } else {
            format!("shaders/spv/token_shift_{dt_i}_{dt_o}.spv")
        };
        let file = Asset::get(&path).ok_or("failed to find shader")?;
        let shader = file.data;

        // specialization constants (matching token_shift.comp)
        // constant_id 0: C (channel size)
        // constant_id 1: T (number of tokens)
        // constant_id 2-4: STRIDE_X_X, STRIDE_X_Y, STRIDE_X_Z
        // constant_id 5-7: STRIDE_S_X, STRIDE_S_Y, STRIDE_S_Z
        // constant_id 8-10: STRIDE_Y_X, STRIDE_Y_Y, STRIDE_Y_Z
        // constant_id 11-13: STRIDE_T_X, STRIDE_T_Y, STRIDE_T_Z
        let specialization = [
            *channel as u32,              // C
            *num_tokens as u32,           // T
            layout_x.stride_of(0) as u32, // STRIDE_X_X
            layout_x.stride_of(1) as u32, // STRIDE_X_Y
            layout_x.stride_of(2) as u32, // STRIDE_X_Z
            layout_s.stride_of(0) as u32, // STRIDE_S_X
            layout_s.stride_of(1) as u32, // STRIDE_S_Y
            layout_s.stride_of(2) as u32, // STRIDE_S_Z
            layout_y.stride_of(0) as u32, // STRIDE_Y_X
            layout_y.stride_of(1) as u32, // STRIDE_Y_Y
            layout_y.stride_of(2) as u32, // STRIDE_Y_Z
            layout_t.stride_of(0) as u32, // STRIDE_T_X
            layout_t.stride_of(1) as u32, // STRIDE_T_Y
            layout_t.stride_of(2) as u32, // STRIDE_T_Z
        ];

        let mode = if *reversed { "reversed" } else { "normal" };
        log::info!("feature: token shift ({mode})");
        log::info!(
            "channel: {channel}, tokens: {num_tokens}, states: {num_states}, batch: {batch}"
        );
        log::info!("\tspecialization: {:?}", specialization);

        let bindings = [vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let kernel = app.create_kernel(&shader, &specialization, &bindings)?;
        kernel.binder().bind_uniform(params, 0, 0).build();

        // dispatch dimensions:
        //   x = ceil(C / BLOCK_SIZE) — one workgroup per channel block
        //   y = T — one workgroup row per token
        //   z = batch
        let block_size = 256u32;
        let dispatch_x = (*channel as u32).div_ceil(block_size);

        unsafe {
            let flags = vk::CommandBufferResetFlags::empty();
            app.device.reset_command_buffer(compute[0], flags)?;

            // transfer output init data to device
            let info = vk::SubmitInfo::builder().command_buffers(&transfer[1..=1]);
            let queue = app.transfer.queue;
            app.device.queue_submit(queue, &[info], vk::Fence::null())?;
            app.device.queue_wait_idle(queue)?;

            // record compute command buffer
            let info = vk::CommandBufferBeginInfo::builder();
            app.device.begin_command_buffer(compute[0], &info)?;
            kernel.cmd_bind(compute[0], &[]);
            for _ in 0..REPEAT {
                app.device
                    .cmd_dispatch(compute[0], dispatch_x, *num_tokens as u32, *batch as u32);
            }
            app.device.end_command_buffer(compute[0])?;

            #[cfg(not(feature = "correctness"))]
            for _ in 0..WARMUP {
                let info = vk::SubmitInfo::builder().command_buffers(compute);
                let queue = app.compute.queue;
                app.device.queue_submit(queue, &[info], vk::Fence::null())?;
                app.device.queue_wait_idle(queue)?;
            }

            // execute compute
            let info = vk::SubmitInfo::builder().command_buffers(compute);
            let queue = app.compute.queue;
            app.device.queue_submit(queue, &[info], vk::Fence::null())?;

            #[cfg(not(feature = "correctness"))]
            {
                let timer = std::time::Instant::now();
                app.device.queue_wait_idle(queue)?;
                let duration = timer.elapsed();

                let ops = 3 * channel * num_tokens * batch * REPEAT; // mix = 3 ops per element
                let duration = duration.as_secs_f64();
                let gflops = (ops as f64) / (duration * 1e9);
                log::info!("\tGFLOPS: {gflops:.3}");
                log::info!("\tduration: {duration:?}, operations: {ops}, repeat: {REPEAT}");
            }
            #[cfg(feature = "correctness")]
            app.device.queue_wait_idle(queue)?;

            // transfer output back to host
            let info = vk::SubmitInfo::builder().command_buffers(&transfer[2..=2]);
            let queue = app.transfer.queue;
            app.device.queue_submit(queue, &[info], vk::Fence::null())?;
            app.device.queue_wait_idle(queue)?;
        }

        #[cfg(feature = "correctness")]
        assert_eq!(self.check_ans(&ans), 0);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use half::f16;
    use simplelog::{LevelFilter, SimpleLogger};

    use super::*;
    use crate::app::App;

    fn init_test() {
        _ = SimpleLogger::init(LevelFilter::Debug, Default::default());
        fastrand::seed(514);
    }

    #[test]
    fn test_token_shift_f16_f16_normal() -> Result<(), Box<dyn Error>> {
        init_test();

        let app = App::new()?;

        let channel = 4096;
        let num_tokens = 64;
        let num_states = 4;
        let batch = 2;

        let bench =
            TokenShiftBench::<f16, f16>::new(&app, channel, num_tokens, num_states, batch, false)?;
        bench.benchmark_token_shift()
    }

    #[test]
    fn test_token_shift_f16_f16_reversed() -> Result<(), Box<dyn Error>> {
        init_test();

        let app = App::new()?;

        let channel = 4096;
        let num_tokens = 64;
        let num_states = 4;
        let batch = 2;

        let bench =
            TokenShiftBench::<f16, f16>::new(&app, channel, num_tokens, num_states, batch, true)?;
        bench.benchmark_token_shift()
    }

    #[test]
    fn test_token_shift_f16_f16_small() -> Result<(), Box<dyn Error>> {
        init_test();

        let app = App::new()?;

        // minimal sizes for basic correctness
        let channel = 8;
        let num_tokens = 4;
        let num_states = 2;
        let batch = 1;

        let bench =
            TokenShiftBench::<f16, f16>::new(&app, channel, num_tokens, num_states, batch, false)?;
        bench.benchmark_token_shift()
    }

    #[test]
    fn test_token_shift_f16_f16_single_state() -> Result<(), Box<dyn Error>> {
        init_test();

        let app = App::new()?;

        // single state: all tokens in one continuous sequence
        let channel = 32;
        let num_tokens = 8;
        let num_states = 1;
        let batch = 1;

        let bench =
            TokenShiftBench::<f16, f16>::new(&app, channel, num_tokens, num_states, batch, false)?;
        bench.benchmark_token_shift()
    }

    #[test]
    fn test_token_shift_f16_f16_multi_batch() -> Result<(), Box<dyn Error>> {
        init_test();

        let app = App::new()?;

        // multiple batches (e.g., processing q/k/v simultaneously)
        let channel = 64;
        let num_tokens = 8;
        let num_states = 2;
        let batch = 8;

        let bench =
            TokenShiftBench::<f16, f16>::new(&app, channel, num_tokens, num_states, batch, false)?;
        bench.benchmark_token_shift()
    }
}
