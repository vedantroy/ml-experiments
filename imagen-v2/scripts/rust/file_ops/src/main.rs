use anyhow::{anyhow, bail, Result};
use glob::glob;
use rio;
use futures::{stream::FuturesUnordered, StreamExt};
use std::fs::{self, File};
use std::path::Path;

// `O_DIRECT` requires all reads and writes
// to be aligned to the block device's block
// size. 4096 might not be the best, or even
// a valid one, for yours!
//#[repr(align(4096))]
//struct Aligned([u8; CHUNK_SIZE as usize]);

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
        .filter_map(|x| x.to_str().map(|x| x.to_owned()))
        .collect();

    let chunk_size = std::cmp::max(10, files.len() / n_workers);
    let chunks = files.chunks(chunk_size);

    //let mut handles = vec![];
    let mut futures = FuturesUnordered::new();

    let ring = rio::new().unwrap();
    for chunk in chunks {
        let ring = ring.clone();
        let chunk = chunk.to_vec();
        let handle = tokio::spawn(async move {
            for path in chunk {
                let f = File::open(path).unwrap();
                let meta = f.metadata().unwrap();
                let file_size = meta.len();
                print!("fsize: {}", file_size);

                let iov = Vec::with_capacity(file_size as usize);
                let mut bytes_read = 0;
                loop {
                    let read = ring.read_at(&f, &iov, bytes_read);
                    bytes_read += read.await.unwrap() as u64;
                    if bytes_read == file_size as u64 {
                        break;
                    } else {
                        println!("partial len: {}", bytes_read);
                    }
                }
                println!("bytes: {}", bytes_read)
            }
        });
        //handles.push(handle);
        futures.push(handle);
    }

    while let x = futures.next() {
        println!("task finished");
    }

    Ok(())
}
