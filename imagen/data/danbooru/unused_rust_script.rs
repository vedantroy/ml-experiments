use anyhow::Result;
use std::{
    fs::{self, File},
    io::{BufReader, BufRead},
};

fn get_json_files(dir: &str) -> Result<Vec<String>> {
    let mut files = vec![];
    let entries: Vec<_> = fs::read_dir(dir)?
        .filter_map(|x| x.ok())
        .filter_map(|x| {
            x.path()
                .to_str()
                .map(|x| x.to_string())
                .zip(x.file_type().ok())
                .zip(x.path().extension().map(|x| x.to_owned()))
        })
        .collect();
    for ((path, file_type), ext) in entries {
        let is_json_file = file_type.is_file() && ext == "json";
        if !is_json_file {
            continue;
        }
        files.push(path);
    }
    Ok(files)
}

fn main() -> Result<()> {
    let files = get_json_files("../big_chunks/")?;
    for file in files {
        let reader = BufReader::new(File::open(file)?);
        for line in reader.lines() {
        }
    }
    println!("{:#?}", files);
    Ok(())
}
