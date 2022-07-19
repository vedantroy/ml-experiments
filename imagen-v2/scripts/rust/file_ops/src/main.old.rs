use anyhow::{anyhow, bail, Result};
use futures::{future::join_all, stream::FuturesUnordered, StreamExt};
use glob::glob;
use rio;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{fs, thread::JoinHandle};
use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncWriteExt},
};
use indicatif::{ProgressBar};

static FILES_WRITTEN: AtomicUsize = AtomicUsize::new(0);

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
    let n_workers = 140;

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

    let chunk_size = std::cmp::max(10, files.len() / n_workers);
    let chunks = files.chunks(chunk_size);

    let mut futures = vec![];



    for chunk in chunks {
        let dest_dir_path = dest_dir_path.clone();
        let chunk = chunk.to_vec();
        let handle = tokio::spawn(async move {
            for path in chunk {
                let buf = tokio::fs::read(&path).await.expect("Could not read file");
                //let mut f = File::open(&path)
                //    .await
                //    .expect("Could not open file for reading");
                //let mut buf = vec![];
                //f.read_buf(&mut buf).await.expect("Could not read file");

                let bucket = path.parent().unwrap();
                let out_path = dest_dir_path
                    .join(bucket.file_name().unwrap())
                    .join(path.file_name().unwrap());

                //println!("Writing to: {:#?}", out_path);
                let mut out = File::create(out_path)
                    .await
                    .expect("Could not open file for writing");
                out.write_all(&buf).await.expect("Could not write to file");

                //FILES_WRITTEN.fetch_add(1, Ordering::Relaxed);
            }
        });
        futures.push(handle);
    }

    join_all(futures).await;
    println!("Files written: {:#?}", FILES_WRITTEN);
    //while let x = futures.next() {
    //    println!("task finished");
    //}

    Ok(())
}
