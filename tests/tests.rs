use candle_core::Device;
use candle_llm_dataset::{LLMDataset, LLMDatasetIter};
use tokenizers::Tokenizer;

fn get_tokenizer() -> Tokenizer {
    let tokenizer_path = {
        let api = hf_hub::api::sync::Api::new().unwrap();
        let api = api.model("hf-internal-testing/llama-tokenizer".to_string());
        api.get("tokenizer.json").unwrap()
    };
    Tokenizer::from_file(tokenizer_path).unwrap()
}

#[test]
fn add_lines() {
    let mut dataset = LLMDataset::new(Vec::new(), Device::Cpu, get_tokenizer());
    dataset
        .add_line(
            "This is a test line.".to_string(),
            false,
            Some("<s>".into()),
            Some("</s>".into()),
        )
        .unwrap();
    assert_eq!(dataset.length(), 1);
    dataset
        .add_line(
            "This is also test line.".to_string(),
            false,
            Some("<s>".into()),
            Some("</s>".into()),
        )
        .unwrap();
    assert_eq!(dataset.length(), 2);
}

#[test]
fn get_next() {
    let mut dataset = LLMDataset::new(Vec::new(), Device::Cpu, get_tokenizer());
    dataset
        .add_line(
            "This is a test line.".to_string(),
            false,
            Some("<s>".into()),
            Some("</s>".into()),
        )
        .unwrap();

    let mut iter = LLMDatasetIter::new_shuffled(&dataset, 1);
    let next = iter.next().unwrap();
    assert!(next.input.attention_mask.is_some());
    assert_eq!(next.input.ids.dim(1).unwrap(), 8);

    assert!(next.target.attention_mask.is_some());
    assert_eq!(next.target.ids.dim(1).unwrap(), 8);
}

#[test]
fn batches() {
    let mut dataset = LLMDataset::new(Vec::new(), Device::Cpu, get_tokenizer());
    dataset
        .add_line(
            "This is test line 1.".to_string(),
            false,
            Some("<s>".into()),
            Some("</s>".into()),
        )
        .unwrap();
    dataset
        .add_line(
            "This is test line 2.".to_string(),
            false,
            Some("<s>".into()),
            Some("</s>".into()),
        )
        .unwrap();
    dataset
        .add_line(
            "This is test line 3.".to_string(),
            false,
            Some("<s>".into()),
            Some("</s>".into()),
        )
        .unwrap();
    dataset
        .add_line(
            "This is test line 4.".to_string(),
            false,
            Some("<s>".into()),
            Some("</s>".into()),
        )
        .unwrap();

    let mut iter = LLMDatasetIter::new_shuffled(&dataset, 2);
    let next = iter.next().unwrap();
    assert_eq!(next.input.ids.dim(0).unwrap(), 2);
    assert_eq!(next.target.ids.dim(0).unwrap(), 2);
    assert_eq!(next.input.attention_mask.unwrap().dim(0).unwrap(), 2);
    assert_eq!(next.target.attention_mask.unwrap().dim(0).unwrap(), 2);
}
