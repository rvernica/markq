//! Shared `Candidate` wire type: the element of the JSON array `markq rerank`
//! reads on stdin (the same shape `search`/`vsearch`/`query --json` emit).
//!
//! Only `id` and `text` are typed; everything else (`score`, `collection`,
//! and any unknown keys) is captured via `#[serde(flatten)]` so it round-trips
//! verbatim without markq needing to know its shape.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    pub text: String,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_passthrough_fields() {
        let input = r#"[{"id":"a","text":"foo","score":1.5,"collection":"docs","extra":true}]"#;

        let candidates: Vec<Candidate> = serde_json::from_str(input).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].id, "a");
        assert_eq!(candidates[0].text, "foo");

        let round_tripped = serde_json::to_value(&candidates[0]).unwrap();
        assert_eq!(round_tripped["score"], serde_json::json!(1.5));
        assert_eq!(round_tripped["collection"], serde_json::json!("docs"));
        assert_eq!(round_tripped["extra"], serde_json::json!(true));
    }
}
