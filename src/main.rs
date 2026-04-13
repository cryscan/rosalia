use std::error::Error;

pub mod app;
pub mod asset;
pub mod benchmark;
pub mod layout;
pub mod num;
pub mod sam;

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

fn main() -> Result<(), Box<dyn Error>> {
    init()?;

    Ok(())
}
