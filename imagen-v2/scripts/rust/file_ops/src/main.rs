use anyhow::{anyhow, bail, Result};
use futures::{future::join_all, stream::FuturesUnordered, StreamExt};
use glob::glob;
use indicatif::ProgressBar;
use rayon;
use rayon::prelude::*;
use std::io::Write;
use std::path::Path;
use std::{
    fs::{self, File},
    thread::JoinHandle,
};

fn glob_to_file_names(pattern: &str) -> Result<Vec<String>> {
    Ok(glob(pattern)?
        .filter_map(|x| x.ok())
        .filter_map(|x| x.file_name().map(|x| x.to_str().map(|x| x.to_string())))
        .filter_map(|x| x)
        .collect())
}

#[tokio::main]
async fn main() -> Result<()> {
    let dest_dir = "../../../data/danbooru/raw/valid_imgs_4";
    let src_dir = "../../../data/danbooru/raw/imgs";

    let src_dir_path = Path::new(src_dir);
    if !src_dir_path.exists() {
        bail!("No source directory: {}", src_dir)
    }

    fs::create_dir_all(dest_dir)?;
    let dest_dir_path = Path::new(dest_dir);

    let buckets = glob_to_file_names(&format!("{}/*", src_dir))?;
    for bucket in buckets {
        let bucket_path = dest_dir_path.join(bucket);
        fs::create_dir_all(
            bucket_path
                .to_str()
                .ok_or(anyhow!("Could not convert path: {:#?} to str", bucket_path))?,
        )?;
    }

    let files: Vec<_> = glob(&format!("{}/*/*", src_dir))?
        .filter_map(|x| x.ok())
        .collect();
    println!("Processing: {} files", files.len());

    rayon::ThreadPoolBuilder::new()
        .num_threads(64)
        .build_global()
        .unwrap();

    let res: Vec<_> = files
        .par_iter()
        .map(|path| {
            let buf = fs::read(&path).expect("Could not read file");
            let bucket = path.parent().unwrap();
            let out_path = dest_dir_path
                .join(bucket.file_name().unwrap())
                .join(path.file_name().unwrap());

            //println!("Writing to: {:#?}", out_path);
            let mut out = File::create(out_path).expect("Could not open file for writing");
            out.write_all(&buf).expect("Could not write to file");
        })
        .collect();

    Ok(())
}
