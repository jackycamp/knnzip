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
    let k = 5;
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
    // let target_klass = 3;
    // let test_case = "Japan’s Bonds Decline on Rate Bets, HK Stocks Gain: Markets Wrap";
    let target_klass = 2;
    let test_case = "Adrian Beltré, Todd Helton and Joe Mauer elected to baseball’s Hall of Fame";
    let target_klass = 2;
    let test_case = "Bucks fire coach Adrian Griffin after 43 games despite having one of NBA’s top records";
    // let target_klass = 4;
    // let test_case = "Private US lander destroyed during reentry after failed mission to moon, company says";
    println!("input text: {}", test_case);
    println!("target class: {}", labels[target_klass - 1]);
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
                dist.push((ncd, train_text));
            }
        });
        handles.push(handle);
    }

    // for i in 0..train_size {
    //     let dist_shared_clone = dist_shared.clone();
    //     let train_set_shared_clone = Arc::clone(&train_set_shared);
    //     let handle = std::thread::spawn(move || {
    //         // let train_set_shared_clone = Arc::clone(&train_set_shared);
    //         let train_label = train_set_shared_clone[i].0.clone();
    //         let train_text = train_set_shared_clone[i].1.clone();
    //
    //         let cx2 = compress(&train_text).len();
    //         let combined = format!("{} {}", test_case, train_text);
    //         let cx1x2 = compress(&combined).len();
    //         let ncd = (cx1x2 as f64 - cx1.min(cx2) as f64) / cx1.max(cx2) as f64;
    //         let mut dist = dist_shared_clone.lock().unwrap();
    //         dist.push((ncd, train_text));
    //     });
    //     handles.push(handle);
    // }

    for handle in handles {
        handle.join().expect("could not join thread");
    }

    let dist = dist_shared.lock().unwrap();

    let mut indices_with_distances: Vec<_> = dist.iter().enumerate().collect();
    indices_with_distances.sort_by(|&(_, dist_a), &(_, dist_b)| dist_a.partial_cmp(dist_b).unwrap());
    let top_classes: Vec<_> = indices_with_distances.iter().take(k).map(|&(idx, _)| train_set_shared_outer[idx].0).collect();
    println!("top classes {:?}", top_classes);

    let mut counts = HashMap::new();
    for class in &top_classes {
        *counts.entry(class).or_insert(0) += 1;
    }

    let predict_class = counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(class, _)| *class)
        .expect("no class found!");
    let predict_class_idx = (predict_class - 1) as usize;
    println!("predicted class: {}, {}", labels[predict_class_idx], predict_class);

}
