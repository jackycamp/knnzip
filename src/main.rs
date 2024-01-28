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
    // println!("data: {:?}", data);
    // println!("len: {:?}", data.len());
    Ok(data)
}


fn main() {
    let k = 10;
    let train_size = 120000;
    let batch_size = 10000;
    // NOTE: original labels are 1, 2, 3, 4
    let labels = ["World", "Sports", "Business", "Sci/Tech"];
    let test_set = read_csv("data/ag_news/test.csv", None).expect("cannot load test set");
    let train_set = read_csv("data/ag_news/train.csv", Some(train_size as usize)).expect("cannot load train set");
    println!("number samples in test set {}", &test_set.len());
    println!("number of samples in train set {}", &train_set.len());
    println!("k: {}, train size: {}", k, train_size);
    println!("batch size: {}", batch_size);
    

    // let test_case = "EBay gets into rentals EBay plans to buy the apartment and home rental service Rent.com for $415 million, adding to its already exhaustive breadth of offerings.";
    let target_klass = 3;
    let test_case = "Japan’s Bonds Decline on Rate Bets, HK Stocks Gain: Markets Wrap";
    // let target_klass = 2;
    // let test_case = "Adrian Beltré, Todd Helton and Joe Mauer elected to baseball’s Hall of Fame";
    // let target_klass = 2;
    // let test_case = "Bucks fire coach Adrian Griffin after 43 games despite having one of NBA’s top records";
    // let target_klass = 4;
    // let test_case = "Private US lander destroyed during reentry after failed mission to moon, company says";
    println!("input text: {}", test_case);
    println!("target class: {}, (idx: {})", labels[target_klass - 1], target_klass);
    let cx1 = compress(test_case).len();
    let train_set_shared = Arc::new(train_set);

    let dist_shared = std::sync::Arc::new(std::sync::Mutex::new(vec![]));
    let mut handles = vec![];

    let train_set_shared_outer = Arc::clone(&train_set_shared);

    for (i, batch) in train_set_shared.chunks(batch_size).enumerate() {
        let dist_shared_clone = Arc::clone(&dist_shared);
        let train_set_shared_clone = Arc::clone(&train_set_shared);

        let start_idx = i * batch_size;
        let end_idx = start_idx + batch.len();

        let handle = std::thread::spawn(move || {
            for idx in start_idx..end_idx {
                let sample = &train_set_shared_clone[idx];
                let train_text = sample.1.clone();
                let cx2 = compress(&train_text).len();
                let combined = format!("{} {}", test_case, train_text);
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
    let top_k_indices: Vec<usize> = dist.iter().take(k).map(|&(idx,_)|idx).collect();
    let mut class_counts: HashMap<&i32, usize> = HashMap::new();
    for &idx in &top_k_indices {
        let rec = &train_set_shared_outer[idx];
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
