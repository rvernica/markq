//! `markq doctor`: one read-only pass that reports the dataset's presence
//! and identity, index coverage, and model cache state, each finding carrying
//! a severity and a single actionable next step.

use serde::Serialize;

/// Severity of one diagnostic finding (data-model.md `Diagnostic`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Ok,
    Warning,
    Error,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Severity::Ok => "ok",
            Severity::Warning => "warning",
            Severity::Error => "error",
        })
    }
}

/// One finding from `markq doctor`. `next_step` is required to be non-empty
/// whenever `severity` is not `Ok` (FR-019); an `Ok` finding may still carry
/// one (e.g. "not yet initialized, run `markq index`" is `Ok`, not an error,
/// per FR-023).
#[derive(Debug, Clone, Serialize)]
pub struct Diagnostic {
    pub check: String,
    pub severity: Severity,
    pub observed: Option<String>,
    pub expected: Option<String>,
    pub next_step: String,
}

impl Diagnostic {
    pub fn new(
        check: impl Into<String>,
        severity: Severity,
        observed: Option<String>,
        expected: Option<String>,
        next_step: impl Into<String>,
    ) -> Self {
        Self {
            check: check.into(),
            severity,
            observed,
            expected,
            next_step: next_step.into(),
        }
    }

    /// A finding with nothing wrong and nothing to do.
    pub fn ok(check: impl Into<String>) -> Self {
        Self::new(check, Severity::Ok, None, None, "")
    }

    pub fn warning(
        check: impl Into<String>,
        observed: impl Into<String>,
        expected: impl Into<String>,
        next_step: impl Into<String>,
    ) -> Self {
        Self::new(
            check,
            Severity::Warning,
            Some(observed.into()),
            Some(expected.into()),
            next_step,
        )
    }

    pub fn error(
        check: impl Into<String>,
        observed: impl Into<String>,
        expected: impl Into<String>,
        next_step: impl Into<String>,
    ) -> Self {
        Self::new(
            check,
            Severity::Error,
            Some(observed.into()),
            Some(expected.into()),
            next_step,
        )
    }
}

/// Whether the process should exit with a failure status: true when any
/// finding is `Severity::Error` (FR-021). Warnings alone do not fail.
pub fn any_error(findings: &[Diagnostic]) -> bool {
    findings.iter().any(|f| f.severity == Severity::Error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn any_error_finding_yields_failure_status() {
        let all_ok = vec![
            Diagnostic::ok("dataset.present"),
            Diagnostic::warning("models.reranker", "unverified", "verified", "run pull"),
        ];
        assert!(
            !any_error(&all_ok),
            "warnings alone must not fail the exit status"
        );

        let with_error = vec![
            Diagnostic::ok("dataset.present"),
            Diagnostic::error("models.embedder", "absent", "verified", "run pull"),
        ];
        assert!(any_error(&with_error));
    }

    #[test]
    fn ok_findings_carry_no_next_step() {
        let d = Diagnostic::ok("dataset.present");
        assert_eq!(d.severity, Severity::Ok);
        assert_eq!(d.next_step, "");
        assert_eq!(d.observed, None);
        assert_eq!(d.expected, None);
    }
}
