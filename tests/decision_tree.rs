#[cfg(test)]
mod decision_tree {
    use stellar::learning::decision_tree::{
        DecisionTree,
        DecisionTreeModel
    }; 
    use std::fs::File;
    use csv::ReaderBuilder;

    // functions
    fn read_boston_data() -> Vec<Vec<f32>> {
        let file = File::open("test_data/boston_housing.csv").unwrap();
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);

        // Read header to know number of columns
        let headers = rdr.headers().unwrap().clone();
        let n_cols = headers.len();

        let mut data: Vec<Vec<f32>> = vec![Vec::new(); n_cols];
        for result in rdr.records() {
            let record = result.unwrap();
            for (i, field) in record.iter().enumerate() {
                let val: f32 = field.parse().unwrap_or(f32::NAN);
                data[i].push(val);
            }
        }
        data
    }
    #[test]
    fn boston_tree_basic_test() {
        // Load data
        let data = read_boston_data();
        let mut dt = DecisionTree::new(&data, 1_f32, 1_f32);

        // Train tree
        let model:DecisionTreeModel = dt.train(8);

        // Check dimensions
        assert_eq!(data.len(), model.metadata[0].dim + 1, "Number of columns mismatch");

        // Test prediction on a known row (e.g., row 32)
        let idx = 32;
        let test_input: Vec<f32> = data.iter().map(|col| col[idx]).collect();
        let prediction = model.predict(&test_input);
        assert!(prediction >= 0.0 && prediction < 100.0, "Prediction out of plausible range");

        // Test variance analysis
        let gains = model.analyze_gains();
        for &g in &gains { assert!(g >= 0.0, "Negative variance gain found"); }

        let total_sse = model.metadata[0].sse();
        let cumulative: f32 = gains.iter().sum();
        println!("cumulative {cumulative}, total_sse {total_sse}");
        assert!(cumulative <= total_sse, "Cumulative variance gain exceeds total SSE");
        assert!(cumulative / total_sse > 0.65, "Total Explained Variance too low -> degredation");
        assert!(cumulative / total_sse < 0.98, "Total Explained Variance too high to be realistic");
    }
    #[test]
    fn decision_tree_subsample() {
        // Load data
        let data = read_boston_data();
        let mut dt = DecisionTree::new(&data, 0.8, 0.8);
        let model:DecisionTreeModel = dt.train(8);
        // Test variance analysis
        let gains = model.analyze_gains();
        for &g in &gains { assert!(g >= 0.0, "Negative variance gain found"); }
        let total_sse = model.metadata[0].sse();
        let cumulative: f32 = gains.iter().sum();
        println!("cumulative {cumulative}, total_sse {total_sse}");
        assert!(cumulative <= total_sse, "Cumulative variance gain exceeds total SSE");
        assert!(cumulative / total_sse > 0.65, "Total Explained Variance too low -> degredation");
        assert!(cumulative / total_sse < 0.98, "Total Explained Variance too high to be realistic");
    }
}

