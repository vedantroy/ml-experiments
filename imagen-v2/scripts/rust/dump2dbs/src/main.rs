use std::fs::Metadata;
use std::path::{Path, PathBuf};
use std::{fs as std_fs, os::unix::prelude::MetadataExt};

use anyhow::{anyhow, bail, Result};
use futures::future;
use glob::glob;
use indicatif::ProgressBar;
use tokio_uring::fs::{self, File};

fn glob_to_file_names(pattern: &str) -> Result<Vec<String>> {
    Ok(glob(pattern)?
        .filter_map(|x| x.ok())
        .filter_map(|x| x.file_name().map(|x| x.to_str().map(|x| x.to_string())))
        .filter_map(|x| x)
        .collect())
}

async fn worker(
    chunk: Vec<(PathBuf, Metadata)>,
    bar: ProgressBar,
    dest_dir: PathBuf,
) -> Result<()> {
    // Pre-allocating & resizing the buffer increases io perf from
    // 2gb/s => 2.35gb/s, but I can't figure it out w/ the borrow-checker
    // let mut buf = vec![];
    for (path, meta) in chunk {
        let target_bytes = meta.size() as usize;
        let mut buf = Vec::with_capacity(target_bytes);
        // if buf.len() < target_bytes {
        //     buf.resize(target_bytes, 0);
        // }
        let mut bytes_read = 0;

        {
            let in_f = match File::open(&path).await {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("Error opening file: {:#?}\n{}", path, e.to_string());
                    continue;
                }
            };
            while bytes_read < target_bytes {
                let out = in_f.read_at(buf, bytes_read as u64).await;
                bytes_read += out.0?;
                buf = out.1;
            }
        }

        let bucket = path
            .parent()
            .ok_or(anyhow!("No parent"))?
            .file_name()
            .ok_or(anyhow!("No file name"))?;
        let file_name = path.file_name().ok_or(anyhow!("No file name"))?;
        let out_path = dest_dir.join(bucket).join(file_name);

        {
            let out_f = match File::create(&out_path).await {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("Error creating file: {:#?}\n{}", out_path, e.to_string());
                    continue;
                }
            };
            let mut bytes_written = 0;
            while bytes_written < target_bytes {
                let out = out_f.write_at(&buf[..target_bytes], bytes_written as u64).await;
                bytes_written += out.0?;
                buf = out.1;
            }
        }

        bar.inc(1);
    }
    Ok(())
}

fn main() -> Result<()> {
    //let dest_dir = "/home/vedant/Desktop/ml-experiments/imagen-v2/data/danbooru/raw/valid_imgs";
    let dest_dir = "./imgs";
    let src_dir = "/home/vedant/Desktop/ml-experiments/imagen-v2/data/danbooru/raw/imgs";
    let n_workers = 12;

    let src_dir_path = Path::new(src_dir);
    if !src_dir_path.exists() {
        bail!("No source directory: {}", src_dir)
    }

    std_fs::create_dir_all(dest_dir)?;
    let dest_dir_path = Path::new(dest_dir);

    let buckets = glob_to_file_names(&format!("{}/*", src_dir))?;
    for bucket in buckets {
        let bucket_path = dest_dir_path.join(bucket);
        std_fs::create_dir_all(
            bucket_path
                .to_str()
                .ok_or(anyhow!("Could not convert path: {:#?} to str", bucket_path))?,
        )?;
    }

    let files: Vec<_> = glob(&format!("{}/*/*", src_dir))?
        .filter_map(|x| x.ok())
        .filter_map(|x| Some(x.clone()).zip(std_fs::File::open(x).ok()))
        .filter_map(|(path, file)| Some(path).zip(file.metadata().ok()))
        .collect();

    let bar = ProgressBar::new(files.len() as u64);

    println!("Processing: {} files", files.len());
    let chunk_size = std::cmp::max(10, files.len() / n_workers);
    let chunks = files.chunks(chunk_size);

    tokio_uring::start(async {
        let mut handles = vec![];

        for chunk in chunks {
            let chunk = chunk.to_vec();
            let bar = bar.clone();
            let dest_dir_path = dest_dir_path.to_path_buf();

            let handle = tokio_uring::spawn(async move {
                worker(chunk, bar, dest_dir_path).await.unwrap();
            });
            handles.push(handle);
        }
        future::join_all(handles).await;

        Ok(())
    })
}
