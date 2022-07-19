use anyhow::{anyhow, bail, Result};
use glob::glob;
//use indicatif::ProgressBar;
use parquet::{
    column::writer::ColumnWriter,
    data_type::ByteArray,
    //data_type::ByteArray,
    file::{properties::WriterProperties, writer::SerializedFileWriter},
    schema::parser::parse_message_type,
};
use rayon::prelude::*;
//use std::io::Write;
use std::{
    fs::{self, File},
    sync::Arc,
};
use std::{
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
};

static GLOBAL_SHARD_COUNT: AtomicUsize = AtomicUsize::new(0);

fn glob_to_file_names(pattern: &str) -> Result<Vec<String>> {
    Ok(glob(pattern)?
        .filter_map(|x| x.ok())
        .filter_map(|x| x.file_name().map(|x| x.to_str().map(|x| x.to_string())))
        .filter_map(|x| x)
        .collect())
}

fn main() -> Result<()> {
    let dest_dir = "../../../data/danbooru/raw/imgs_shards";
    let src_dir = "../../../data/danbooru/raw/imgs";

    let src_dir_path = Path::new(src_dir);
    if !src_dir_path.exists() {
        bail!("No source directory: {}", src_dir)
    }

    fs::create_dir_all(dest_dir)?;
    // let dest_dir_path = Path::new(dest_dir);

    //let buckets = glob_to_file_names(&format!("{}/*", src_dir))?;
    //for bucket in buckets {
    //    let bucket_path = dest_dir_path.join(bucket);
    //    fs::create_dir_all(
    //        bucket_path
    //            .to_str()
    //            .ok_or(anyhow!("Could not convert path: {:#?} to str", bucket_path))?,
    //    )?;
    //}

    let files: Vec<_> = glob(&format!("{}/*/*", src_dir))?
        .filter_map(|x| x.ok())
        .collect();
    println!("Processing: {} files", files.len());

    rayon::ThreadPoolBuilder::new()
        .num_threads(24)
        .build_global()
        .unwrap();

    let schema = r#"
        message schema {
            REQUIRED INT32 id;
            REQUIRED BINARY img;
        };
    "#;

    let schema = Arc::new(parse_message_type(schema).unwrap());
    let props = Arc::new(WriterProperties::builder().build());

    let total_chunks = files.len() / 500;
    println!("Total chunks: {}", total_chunks);

    let _: Vec<_> = files
        .par_chunks(500)
        .map(|paths| {
            let shard_idx = GLOBAL_SHARD_COUNT.fetch_add(1, Ordering::Relaxed);
            println!("Writing shard: {}", shard_idx);
            let mut ids = vec![];
            let mut bufs: Vec<ByteArray> = vec![];

            for path in paths {
                let buf = fs::read(&path).expect("Could not read file");
                bufs.push(buf.into());

                // prefix is technically better, but that's unstable
                let id = path.file_stem().unwrap().to_str().unwrap();
                let id = id.parse::<i32>().unwrap();
                ids.push(id);

                // let bucket = path.parent().unwrap();
                // let out_path = dest_dir_path
                //     .join(bucket.file_name().unwrap())
                //     .join(path.file_name().unwrap());

                //println!("Writing to: {:#?}", out_path);
                //let mut out = File::create(out_path).expect("Could not open file for writing");
                //out.write_all(&buf).expect("Could not write to file");
            }

            let mut writer = SerializedFileWriter::new(
                File::create(format!("{}/{}.parquet", dest_dir, shard_idx)).unwrap(),
                (&schema).clone(),
                (&props).clone(),
            )
            .unwrap();
            let mut row_group_writer = writer.next_row_group().unwrap();
            {
                let mut id_writer = row_group_writer.next_column().unwrap().unwrap();
                match id_writer.untyped() {
                    ColumnWriter::Int32ColumnWriter(ref mut typed) => {
                        typed.write_batch(&ids, None, None)
                    }
                    _ => {
                        unimplemented!()
                    }
                }
                .unwrap();
                id_writer.close().unwrap();

                //let mut tag_writer = row_group_writer.next_column().unwrap().unwrap();
                //match tag_writer.untyped() {
                //    ColumnWriter::ByteArrayColumnWriter(ref mut typed) => {
                //        typed.write_batch(&vec![], None, None)
                //    }
                //    _ => {
                //        unimplemented!()
                //    }
                //}
                //.unwrap();
                //tag_writer.close().unwrap();

                let mut img_writer = row_group_writer.next_column().unwrap().unwrap();
                match img_writer.untyped() {
                    ColumnWriter::ByteArrayColumnWriter(ref mut typed) => {
                        typed.write_batch(&bufs, None, None)
                    }
                    _ => {
                        unimplemented!()
                    }
                }
                .unwrap();
                img_writer.close().unwrap();
            }
            row_group_writer.close().unwrap();
            writer.close().unwrap();
        })
        .collect();

    Ok(())
}
