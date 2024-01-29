use std::io::Write;
use flate2::write::GzEncoder;
use flate2::Compression;
use csv::ReaderBuilder;
use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::sync::{Arc, Mutex};
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

// poor-man's arg parser
#[derive(Debug)]
struct Args {
    k: usize,
    train_samples: i32,
    batch_size: usize,
    train_path: String,
    test_path: Option<String>,
    test_sample: String,
    num_test_samples: usize,
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
            help: false,
            num_test_samples: 1000,
        }
    }

    pub fn new() -> Self {
        let env_args: Vec<String> = std::env::args().skip(1).collect();
        let mut args = Self::default();

        for (index, argument) in env_args.iter().enumerate() {
            if "--help" == argument || "-h" == argument {
                args.help = true;
            }
            if "--train-path" == argument {
                args.train_path = env_args[index + 1].to_string();
            }
            if "--test-path" == argument {
                args.test_path = Some(env_args[index + 1].to_string());
            }
            if "--k" == argument {
                args.k = env_args[index + 1].parse::<usize>().expect("cannot parse arg as i32");
            }
            if "--test-sample" == argument {
                args.test_sample = env_args[index + 1].to_string();
            }
            if "--num-test-samples" == argument {
                args.num_test_samples = env_args[index + 1].parse::<usize>().expect("cannot parse arg as usize");
            }
            if "--batch-size" == argument {
                args.batch_size = env_args[index + 1].parse::<usize>().expect("cannot parse arg as usize");
            }
        }
        args
    }

    pub fn print_help(&self) {
        println!("knnzip");
        println!("usage: knnzip [OPTIONS]");
        println!("--train-path: Path to a training dataset (must be a csv)");
        println!("--test-path: Path to a test dataset (must be a csv)");
        println!("--k: number of samples to take when performing k-nearest-neighbors");
        println!("--test-sample: a string to perform text classification on");
        println!("--num-test-samples: number of test samples to consider when in eval mode");
        println!("--batch-size: size of training batch\n");
        println!("example:");
        println!("knnzip --test-sample \"Earth's Forces Are Causing This Massive Plate to Split in Two.\"\n");
        println!("if --test-path is provided, then the script will be executed in eval mode");
        println!("otherwise, it will execute on a single prediction sample");
    }
}

fn predict_single(txt: &str, train_set: Arc<Vec<(i32, String)>>, batch_size: usize, k: usize) -> i32 {
    let cx1 = compress(txt).len();
    let distances_arc = Arc::new(Mutex::new(vec![]));
    let mut handles = vec![];

    for (i, batch) in train_set.chunks(batch_size).enumerate() {
        let dist_arc_clone = Arc::clone(&distances_arc);
        let train_set_arc_clone = Arc::clone(&train_set);
        let start_idx = i * batch_size;
        let end_idx = start_idx + batch.len();
        let test_sample = txt.to_owned();

        let handle = std::thread::spawn(move || {
            for idx in start_idx..end_idx {
                let train_sample = &train_set_arc_clone[idx];
                let train_text = train_sample.1.clone();
                let cx2 = compress(&train_text).len();
                let combined = format!("{} {}", test_sample, train_text);
                let cx1x2 = compress(&combined).len();
                let ncd = (cx1x2 as f64 - cx1.min(cx2) as f64) / cx1.max(cx2) as f64;
                let mut distances = dist_arc_clone.lock().unwrap();
                distances.push((idx, ncd));
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("could not join thread");
    }

    let mut distances = distances_arc.lock().unwrap();
    distances.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
    let top_k_indices: Vec<usize> = distances.iter().take(k).map(|&(idx,_)|idx).collect();
    let mut class_counts: HashMap<&i32, usize> = HashMap::new();
    for &idx in &top_k_indices {
        let class = &train_set[idx].0;
        *class_counts.entry(class).or_insert(0) += 1;
    }

    let predict_class = class_counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1))
        .map(|(class, _)| class);

    predict_class.expect("no predicted class!").to_owned()
}


fn main() {
    let args = Args::new();
    if args.help {
        args.print_help();
        return;
    }
    // NOTE: original labels are 1, 2, 3, 4
    let labels = ["World", "Sports", "Business", "Sci/Tech"];
    let train_set = read_csv("data/ag_news/train.csv", Some(args.train_samples as usize)).expect("cannot load train set");
    let train_set_arc = Arc::new(train_set);

    if let Some(test_path) = args.test_path {
        println!("eval mode");
        let mut correct = 0;
        let test_set = read_csv(&test_path, Some(args.num_test_samples)).expect("cannot load test set");
        let num_samples = test_set.len();
        for (i, sample) in test_set.iter().enumerate() {
            let target_class = sample.0;
            let train_set_arc_clone = Arc::clone(&train_set_arc);
            let predicted_class = predict_single(&sample.1, train_set_arc_clone, args.batch_size, args.k);
            if predicted_class == target_class {
                correct += 1;
            }
            println!("{}/{} target class {}, predicted {}", i+1, num_samples, target_class, predicted_class);
        }
        println!("correct predictions: {}/{}", correct, num_samples);
        println!("accuracy: {}", (correct as f64 / num_samples as f64));
        return;
    }
    println!("single prediction mode");
    let predicted_class = predict_single(&args.test_sample, train_set_arc, args.batch_size, args.k);
    let predicted_class_idx = predicted_class - 1;
    println!("predicted class: {} (class idx {})", labels[predicted_class_idx as usize], predicted_class);
}
