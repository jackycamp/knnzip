use std::io::Write;
use flate2::write::GzEncoder;
use flate2::Compression;
use csv::ReaderBuilder;
use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::sync::Arc;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct Record {
    label: i32,
    title: String,
    desc: String,
}

fn compress(data: &str) -> Vec<u8> {
    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(data.as_bytes()).unwrap();
    e.finish().unwrap()
}

fn read_csv(path: &str, limit: Option<usize>) -> Result<Vec<(i32, String)>, Box<dyn Error>> {
    let file = File::open(path)?;
    println!("reading csv: {}", path);
    let limit = limit.unwrap_or(1000);
    
    let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);
    let mut data: Vec<(i32, String)> = Vec::new();

    for result in rdr.deserialize() {
        if data.len() == limit {
            return Ok(data);
        }
        match result {
            Ok(_) => {
                let record: Record = result?;
                data.push((record.label, format!("{},{}", record.title, record.desc)));
            },
            Err(e) => println!("{}", e),
        }
    }
    Ok(data)
}

#[derive(Debug)]
struct Args {
    k: usize,
    train_samples: i32,
    batch_size: usize,
    train_path: String,
    test_path: Option<String>,
    test_sample: String,
    help: bool,
}

impl Args {
    pub fn default() -> Self {
        Self {
            k: 10,
            train_samples: 120000,
            batch_size: 10000,
            train_path: "data/ag_news/train.csv".to_string(),
            test_path: None,
            test_sample: "2 Magnificent Artificial Intelligence (AI) Growth Stocks Set to Join Apple and Microsoft in the $3 Trillion Club by 2030".to_string(),
            help: false
        }
    }

    pub fn new() -> Self {
        let env_args: Vec<String> = std::env::args().skip(1).collect();
        let mut args = Self::default();

        for (index, argument) in env_args.iter().enumerate() {
            if "--help" == argument || "-h" == argument {
                args.help = true;
            }
            if "--train_path" == argument {
                args.train_path = env_args[index + 1].to_string();
            }
            if "--test_path" == argument {
                args.test_path = Some(env_args[index + 1].to_string());
            }
            if "--k" == argument {
                args.k = env_args[index + 1].parse::<usize>().expect("cannot parse");
            }
            println!("Arg: {}: {}", index + 1, argument);
        }
        args
    }

    pub fn print_help(&self) {
        println!("knnzip");
        println!("usage: knnzip [OPTIONS]");
        println!("for example: knnzip --test_sample \"Earth's Forces Are Causing This Massive Plate to Split in Two.\"");
        println!("--train_path - Path to a training dataset (must be a csv)");
        println!("--test_path - Path to a test dataset (must be a csv)");
    }
}


fn main() {
    let args = Args::new();
    if args.help {
        args.print_help();
        return;
    }
    println!("args: {:?}", args);
    // NOTE: original labels are 1, 2, 3, 4
    let labels = ["World", "Sports", "Business", "Sci/Tech"];
    let test_set = read_csv("data/ag_news/test.csv", None).expect("cannot load test set");
    let train_set = read_csv("data/ag_news/train.csv", Some(args.train_samples as usize)).expect("cannot load train set");
    println!("number samples in test set {}", &test_set.len());
    println!("number of samples in train set {}", &train_set.len());
    println!("k: {}, num train samples: {}", args.k, args.train_samples);
    println!("batch size: {}", args.batch_size);

    let cx1 = compress(&args.test_sample).len();
    let train_set_shared = Arc::new(train_set);

    let dist_shared = std::sync::Arc::new(std::sync::Mutex::new(vec![]));
    let mut handles = vec![];

    let train_set_shared_outer = Arc::clone(&train_set_shared);

    for (i, batch) in train_set_shared.chunks(args.batch_size).enumerate() {
        let test_sample = args.test_sample.clone();
        let dist_shared_clone = Arc::clone(&dist_shared);
        let train_set_shared_clone = Arc::clone(&train_set_shared);

        let start_idx = i * args.batch_size;
        let end_idx = start_idx + batch.len();

        let handle = std::thread::spawn(move || {
            for idx in start_idx..end_idx {
                let sample = &train_set_shared_clone[idx];
                let train_text = sample.1.clone();
                let cx2 = compress(&train_text).len();
                let combined = format!("{} {}", test_sample, train_text);
                let cx1x2 = compress(&combined).len();
                let ncd = (cx1x2 as f64 - cx1.min(cx2) as f64) / cx1.max(cx2) as f64;
                let mut dist = dist_shared_clone.lock().unwrap();
                dist.push((idx, ncd))
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("could not join thread");
    }

    let mut dist = dist_shared.lock().unwrap();
    dist.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
    let top_k_indices: Vec<usize> = dist.iter().take(args.k).map(|&(idx,_)|idx).collect();
    let mut class_counts: HashMap<&i32, usize> = HashMap::new();
    for &idx in &top_k_indices {
        let class = &train_set_shared_outer[idx].0;
        *class_counts.entry(class).or_insert(0) += 1;
    }

    let predict_class = class_counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1))
        .map(|(class, _)| class);
    
    let predict_class_idx = predict_class.expect("no predicted class!").to_owned() as usize;

    println!("predicted class: {} (class idx {})", labels[predict_class_idx - 1], predict_class_idx);
}
