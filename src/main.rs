use std::error::Error;

use clap::{Parser, Subcommand, ValueEnum};
use half::f16;

pub mod app;
pub mod asset;
pub mod benchmark;
pub mod layout;
pub mod num;
pub mod sam;

#[cfg(feature = "correctness")]
const M: usize = 256;
#[cfg(feature = "correctness")]
const N: usize = 256;
#[cfg(not(feature = "correctness"))]
const M: usize = 4096;
#[cfg(not(feature = "correctness"))]
const N: usize = 4096;
const K: usize = 4096;

#[derive(Clone, Debug, ValueEnum)]
enum Precision {
    /// Use f16 precision.
    F16,
    /// Use f32 precision.
    F32,
    /// Run for all precisions.
    All,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run matrix multiplication benchmark.
    Matmul {
        /// Precision to use for matrix multiplication.
        #[arg(long, short, value_enum, default_value_t = Precision::All)]
        precision: Precision,
    },
}

fn init() -> Result<(), Box<dyn Error>> {
    use simplelog::{ColorChoice, CombinedLogger, LevelFilter, TermLogger, WriteLogger};

    std::fs::create_dir_all("logs")?;
    std::fs::create_dir_all("outputs")?;

    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let filename = format!("logs/rosalia_{}.log", timestamp);
    let file = std::fs::File::create(&filename)?;

    CombinedLogger::init(vec![
        TermLogger::new(
            LevelFilter::Debug,
            Default::default(),
            Default::default(),
            ColorChoice::Auto,
        ),
        WriteLogger::new(LevelFilter::Info, Default::default(), file),
    ])?;

    fastrand::seed(514);

    Ok(())
}

/// Run f16 matrix multiplication benchmark.
fn benchmark_matmul_f16(app: &app::App) -> Result<(), Box<dyn Error>> {
    let bench = benchmark::GemmBench::<f16, f16, f16, f16>::new(app, M, N, K)?;
    bench.benchmark_cooperative_matrix()?;
    Ok(())
}

/// Run f32 matrix multiplication benchmark.
fn benchmark_matmul_f32(app: &app::App) -> Result<(), Box<dyn Error>> {
    let bench = benchmark::GemmBench::<f16, f16, f32, f32>::new(app, M, N, K)?;
    bench.benchmark_cooperative_matrix()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    init()?;

    let cli = Cli::parse();
    let app = app::App::new()?;

    match cli.command {
        Command::Matmul { precision } => match precision {
            Precision::F16 => benchmark_matmul_f16(&app)?,
            Precision::F32 => benchmark_matmul_f32(&app)?,
            Precision::All => {
                benchmark_matmul_f16(&app)?;
                benchmark_matmul_f32(&app)?;
            }
        },
    }

    Ok(())
}
