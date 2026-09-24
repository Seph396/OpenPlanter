/// California DRE (Department of Real Estate) public license lookup — free,
/// no API key required. Two endpoints on www2.dre.ca.gov/PublicASP/pplinfo.asp:
///
/// - `GET  pplinfo.asp?License_id=<id>`         — single licensee detail page.
/// - `POST pplinfo.asp?start=1` (LICENSEE_NAME) — name search results table.
///
/// The HTML is old-school ASP table markup (no CSS classes to key off of),
/// so both are parsed by stripping tags to a flat line list and then reading
/// fields off known "Label:" lines — see `strip_to_lines` and
/// `parse_license_page`/`parse_name_search`. Verified against live pages
/// captured 2026-09-23 (fixtures under `tests/fixtures/dre/`).
use serde::Serialize;
use serde_json::json;

use super::ToolResult;

const BASE_URL: &str = "https://www2.dre.ca.gov/PublicASP/pplinfo.asp";
const USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36";

/// Field labels that appear alone on their own line once the HTML is
/// stripped to text (see `strip_to_lines`). Used to find block boundaries
/// on the single-licensee detail page.
const KNOWN_LABELS: &[&str] = &[
    "License Type:",
    "Name:",
    "Mailing Address:",
    "License ID:",
    "Expiration Date:",
    "License Status:",
    "Former Name(s):",
    "Responsible Broker:",
    "Former Responsible Broker:",
    "Main Office:",
    "DBA",
    "Branches:",
    "Affiliated Licensed Corporation(s):",
    "Licensed Officer(s):",
    "Comment:",
];

const END_MARKER: &str = ">>>> Public information request complete";

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LicenseRef {
    pub id: String,
    pub name: String,
    /// e.g. "Officer Expiration Date: 05/06/28" trimmed to just the date.
    pub expiration: String,
    /// Free-text note line following the entry, if any (e.g. "OFFICER
    /// LICENSE EXPIRED AS OF 08/25/04", or a "DESIGNATED OFFICER" header).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Default)]
pub struct LicenseRecord {
    pub license_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expiration_date: Option<String>,
    /// e.g. "03/25/83 (Unofficial -- taken from secondary records)" — from
    /// the "<Type> License Issued:" line, whichever type applies.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issued: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mailing_address: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub former_names: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub responsible_broker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub former_responsible_broker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub main_office: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dbas: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branches: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub affiliated_corporations: Vec<LicenseRef>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub licensed_officers: Vec<LicenseRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disciplinary_comment: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct NameSearchMatch {
    pub license_id: String,
    pub name: String,
    pub license_type: String,
    pub city: String,
}

// ─── HTML → flat line list ───

/// Strip an HTML page down to a list of trimmed, whitespace-collapsed lines,
/// with `<br/>`/`<br>` treated as line breaks and runs of blank lines
/// collapsed to a single blank line. This turns the DRE's ASP table markup
/// into a flat "Label:\nvalue\n\nLabel:\nvalue" structure that's simple to
/// scan for known field labels.
fn strip_to_lines(html: &str) -> Vec<String> {
    let br_re = regex::Regex::new(r"(?i)<br\s*/?>").unwrap();
    let tag_re = regex::Regex::new(r"<[^>]+>").unwrap();
    let ws_re = regex::Regex::new(r"[ \t]+").unwrap();

    let text = br_re.replace_all(html, "\n");
    let text = tag_re.replace_all(&text, "");
    let text = decode_entities(&text);

    let mut out = Vec::new();
    let mut prev_blank = false;
    for line in text.split('\n') {
        let collapsed = ws_re.replace_all(line, " ");
        let trimmed = collapsed.trim().to_string();
        if trimmed.is_empty() {
            if prev_blank {
                continue;
            }
            prev_blank = true;
        } else {
            prev_blank = false;
        }
        out.push(trimmed);
    }
    out
}

fn decode_entities(text: &str) -> String {
    text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
}

/// Find the index of a `KNOWN_LABELS` entry, or a dynamic "<Type> License
/// Issued:" line, at `lines[i]`. Returns the label to key the block under
/// ("License Issued:" for the dynamic case, since the type varies).
fn label_at(lines: &[String], i: usize) -> Option<&'static str> {
    let line = lines[i].as_str();
    for &label in KNOWN_LABELS {
        if line == label {
            return Some(label);
        }
    }
    if line.ends_with("License Issued:") {
        return Some("License Issued:");
    }
    None
}

/// Split `lines` into `(label, block_lines)` pairs, where each block runs
/// from just after its label line to just before the next label line (or
/// end of input). Leading/trailing blank lines in each block are trimmed;
/// blank lines *within* a block (e.g. separating list entries) are kept.
fn label_blocks(lines: &[String]) -> Vec<(&'static str, Vec<String>)> {
    let mut starts: Vec<(usize, &'static str)> = Vec::new();
    for (i, _) in lines.iter().enumerate() {
        if let Some(label) = label_at(lines, i) {
            starts.push((i, label));
        }
    }

    let mut blocks = Vec::new();
    for (idx, &(start, label)) in starts.iter().enumerate() {
        let end = starts.get(idx + 1).map(|&(e, _)| e).unwrap_or(lines.len());
        let mut block: Vec<String> = lines[start + 1..end].to_vec();
        while block.first().is_some_and(|l| l.is_empty()) {
            block.remove(0);
        }
        while block.last().is_some_and(|l| l.is_empty()) {
            block.pop();
        }
        // Trailing boilerplate ("NO OTHER PUBLIC COMMENTS", end marker) only
        // ever trails the last (Comment:) block — drop lines from the end
        // marker onward.
        if let Some(marker_idx) = block.iter().position(|l| l.starts_with(END_MARKER)) {
            block.truncate(marker_idx);
            while block.last().is_some_and(|l| l.is_empty()) {
                block.pop();
            }
        }
        blocks.push((label, block));
    }
    blocks
}

fn block_for<'a>(blocks: &'a [(&'static str, Vec<String>)], label: &str) -> Option<&'a [String]> {
    blocks
        .iter()
        .find(|(l, _)| *l == label)
        .map(|(_, b)| b.as_slice())
}

/// The "Comment:" block spans the disciplinary-action line plus a trailing
/// "public comments" sub-row with no label of its own (see `label_blocks`).
/// Drop the two standard "nothing to report" placeholders and any blank
/// lines; whatever's left (if anything) is real disciplinary/public comment
/// text.
fn extract_comment(block: Option<&[String]>) -> Option<String> {
    let lines = block?;
    let real: Vec<&String> = lines
        .iter()
        .filter(|l| {
            !l.is_empty() && *l != "NO DISCIPLINARY ACTION" && *l != "NO OTHER PUBLIC COMMENTS"
        })
        .collect();
    if real.is_empty() {
        return None;
    }
    Some(real.into_iter().cloned().collect::<Vec<_>>().join("; "))
}

/// A block value that's just prose (no sentinel "NO ..." placeholder) is
/// joined with ", "; DRE's "NO CURRENT ..." / "NO ..." placeholders are
/// treated as absent (`None`).
fn joined_or_none(block: Option<&[String]>) -> Option<String> {
    let lines = block?;
    if lines.is_empty() {
        return None;
    }
    if lines.len() == 1 && lines[0].starts_with("NO ") {
        return None;
    }
    Some(lines.join(", "))
}

/// Parse a "<id> - [Officer ]Expiration Date: <date>" / name / [note] list
/// block (used for both Affiliated Licensed Corporation(s) and Licensed
/// Officer(s) sections — same shape, different wording of the anchor line).
fn parse_license_ref_list(block: &[String]) -> Vec<LicenseRef> {
    let anchor_re = regex::Regex::new(r"^(\d+)\s*-\s*(?:Officer\s+)?Expiration Date:\s*(.+)$").unwrap();

    let mut refs = Vec::new();
    let mut i = 0;
    while i < block.len() {
        if let Some(caps) = anchor_re.captures(&block[i]) {
            let id = caps[1].to_string();
            let expiration = caps[2].trim().to_string();
            let name = block.get(i + 1).cloned().unwrap_or_default();
            let mut note = None;
            let mut consumed = 1; // name line
            if let Some(next) = block.get(i + 2) {
                if !next.is_empty() && !anchor_re.is_match(next) {
                    note = Some(next.clone());
                    consumed = 2;
                }
            }
            refs.push(LicenseRef {
                id,
                name,
                expiration,
                note,
            });
            i += 1 + consumed;
        } else {
            i += 1;
        }
    }
    refs
}

/// Parse a single-licensee detail page
/// (`pplinfo.asp?License_id=<id>`) into a `LicenseRecord`.
pub fn parse_license_page(html: &str) -> Option<LicenseRecord> {
    let lines = strip_to_lines(html);
    let blocks = label_blocks(&lines);
    if blocks.is_empty() {
        return None;
    }

    let license_id = joined_or_none(block_for(&blocks, "License ID:"))?;

    Some(LicenseRecord {
        license_id,
        name: joined_or_none(block_for(&blocks, "Name:")),
        license_type: joined_or_none(block_for(&blocks, "License Type:")),
        status: joined_or_none(block_for(&blocks, "License Status:")),
        expiration_date: joined_or_none(block_for(&blocks, "Expiration Date:")),
        issued: joined_or_none(block_for(&blocks, "License Issued:")),
        mailing_address: joined_or_none(block_for(&blocks, "Mailing Address:")),
        former_names: joined_or_none(block_for(&blocks, "Former Name(s):")),
        responsible_broker: joined_or_none(block_for(&blocks, "Responsible Broker:")),
        former_responsible_broker: joined_or_none(block_for(&blocks, "Former Responsible Broker:")),
        main_office: joined_or_none(block_for(&blocks, "Main Office:")),
        dbas: joined_or_none(block_for(&blocks, "DBA")),
        branches: joined_or_none(block_for(&blocks, "Branches:")),
        affiliated_corporations: block_for(&blocks, "Affiliated Licensed Corporation(s):")
            .map(parse_license_ref_list)
            .unwrap_or_default(),
        licensed_officers: block_for(&blocks, "Licensed Officer(s):")
            .map(parse_license_ref_list)
            .unwrap_or_default(),
        disciplinary_comment: extract_comment(block_for(&blocks, "Comment:")),
    })
}

/// Parse a name-search results page
/// (`POST pplinfo.asp?start=1`, LICENSEE_NAME=...) into result rows.
pub fn parse_name_search(html: &str) -> Vec<NameSearchMatch> {
    let row_re = regex::Regex::new(
        r#"(?is)<a\s+href="pplinfo\.asp\?License_id=(\d+)"\s*>\s*\d+\s*</a>\s*</td>\s*<td>\s*([^<]*?)\s*</td>\s*<td>\s*([^<]*?)\s*</td>\s*<td>\s*([^<]*?)\s*</td>"#,
    )
    .unwrap();

    row_re
        .captures_iter(html)
        .map(|caps| NameSearchMatch {
            license_id: caps[1].to_string(),
            name: decode_entities(caps[2].trim()),
            license_type: decode_entities(caps[3].trim()),
            city: decode_entities(caps[4].trim()),
        })
        .collect()
}

// ─── Tool entry point ───

fn license_ref_json(r: &LicenseRef) -> serde_json::Value {
    json!({
        "id": r.id,
        "name": r.name,
        "expiration": r.expiration,
        "note": r.note,
    })
}

fn record_to_json(record: &LicenseRecord, source_url: &str, fetched_at: &str) -> serde_json::Value {
    let mut out = json!({
        "license_id": record.license_id,
        "name": record.name,
        "license_type": record.license_type,
        "status": record.status,
        "expiration_date": record.expiration_date,
        "issued": record.issued,
        "mailing_address": record.mailing_address,
        "former_names": record.former_names,
        "responsible_broker": record.responsible_broker,
        "former_responsible_broker": record.former_responsible_broker,
        "main_office": record.main_office,
        "dbas": record.dbas,
        "branches": record.branches,
        "disciplinary_comment": record.disciplinary_comment,
        "fetched_at": fetched_at,
        "source_url": source_url,
    });
    if !record.affiliated_corporations.is_empty() {
        out["affiliated_corporations"] = json!(record
            .affiliated_corporations
            .iter()
            .map(license_ref_json)
            .collect::<Vec<_>>());
    }
    if !record.licensed_officers.is_empty() {
        out["licensed_officers"] = json!(record
            .licensed_officers
            .iter()
            .map(license_ref_json)
            .collect::<Vec<_>>());
    }
    out
}

/// `dre_lookup(license_id?, name?, city?)` — look up a CA DRE real estate
/// license by license number or by name ("Last, First"). Free, no API key.
pub async fn dre_lookup(license_id: Option<&str>, name: Option<&str>, city: Option<&str>) -> ToolResult {
    let license_id = license_id.map(str::trim).filter(|s| !s.is_empty());
    let name = name.map(str::trim).filter(|s| !s.is_empty());

    match (license_id, name) {
        (Some(_), Some(_)) => {
            return ToolResult::error(
                "dre_lookup requires exactly one of license_id or name, not both".into(),
            )
        }
        (None, None) => {
            return ToolResult::error("dre_lookup requires either license_id or name".into())
        }
        _ => {}
    }

    let client = reqwest::Client::new();
    let fetched_at = chrono::Utc::now().to_rfc3339();

    if let Some(id) = license_id {
        let url = format!("{BASE_URL}?License_id={id}");
        let resp = client
            .get(&url)
            .header("User-Agent", USER_AGENT)
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await;
        let resp = match resp {
            Ok(r) => r,
            Err(e) => return ToolResult::error(format!("dre_lookup request failed: {e}")),
        };
        if !resp.status().is_success() {
            let status = resp.status();
            return ToolResult::error(format!("dre_lookup request failed ({status})"));
        }
        let body = match resp.text().await {
            Ok(b) => b,
            Err(e) => return ToolResult::error(format!("dre_lookup response read failed: {e}")),
        };
        let record = match parse_license_page(&body) {
            Some(r) => r,
            None => {
                return ToolResult::error(format!(
                    "dre_lookup: no license record found for License_id={id} (check the ID; DRE returns an empty page for unknown IDs)"
                ))
            }
        };
        return ToolResult::ok(
            serde_json::to_string_pretty(&record_to_json(&record, &url, &fetched_at))
                .unwrap_or_default(),
        );
    }

    let name = name.unwrap();
    let form_resp = client
        .get(BASE_URL)
        .header("User-Agent", USER_AGENT)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await;
    let form_body = match form_resp {
        Ok(r) => r.text().await.unwrap_or_default(),
        Err(e) => return ToolResult::error(format!("dre_lookup form fetch failed: {e}")),
    };
    let next_step = regex::Regex::new(r#"(?i)name="h_nextstep"\s+type="hidden"\s+value="([^"]*)""#)
        .ok()
        .and_then(|re| re.captures(&form_body).map(|c| c[1].to_string()))
        .unwrap_or_else(|| "SEARCH".to_string());

    let search_url = format!("{BASE_URL}?start=1");
    let params = [
        ("h_nextstep", next_step.as_str()),
        ("LICENSEE_NAME", name),
        ("CITY_STATE", city.unwrap_or("")),
        ("LICENSE_ID", ""),
    ];
    let resp = client
        .post(&search_url)
        .header("User-Agent", USER_AGENT)
        .timeout(std::time::Duration::from_secs(30))
        .form(&params)
        .send()
        .await;
    let resp = match resp {
        Ok(r) => r,
        Err(e) => return ToolResult::error(format!("dre_lookup name search failed: {e}")),
    };
    if !resp.status().is_success() {
        let status = resp.status();
        return ToolResult::error(format!("dre_lookup name search failed ({status})"));
    }
    let body = match resp.text().await {
        Ok(b) => b,
        Err(e) => return ToolResult::error(format!("dre_lookup response read failed: {e}")),
    };
    let matches = parse_name_search(&body);
    let output = json!({
        "query": name,
        "matches": matches.iter().map(|m| json!({
            "license_id": m.license_id,
            "name": m.name,
            "license_type": m.license_type,
            "city": m.city,
        })).collect::<Vec<_>>(),
        "fetched_at": fetched_at,
        "source_url": search_url,
    });
    ToolResult::ok(serde_json::to_string_pretty(&output).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> String {
        std::fs::read_to_string(format!(
            "{}/tests/fixtures/dre/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap_or_else(|e| panic!("failed to read fixture {name}: {e}"))
    }

    // ── 00000101: Officer, affiliated corps, no own status/expiration ──

    #[test]
    fn test_parse_officer_with_affiliated_corporations() {
        let html = fixture("officer_with_affiliations.html");
        let record = parse_license_page(&html).expect("should parse");

        assert_eq!(record.license_id, "00000101");
        assert_eq!(record.name.as_deref(), Some("Doe, John Quincy"));
        assert_eq!(record.license_type.as_deref(), Some("OFFICER"));
        assert_eq!(record.mailing_address, None); // "NO MAILING ADDRESS"
        assert_eq!(record.status, None); // officer record has no own status line
        assert_eq!(record.former_names, None); // "NO FORMER NAMES"
        assert_eq!(record.disciplinary_comment, None); // "NO DISCIPLINARY ACTION"

        assert_eq!(record.affiliated_corporations.len(), 2);
        let corp1 = &record.affiliated_corporations[0];
        assert_eq!(corp1.id, "00000103");
        assert_eq!(corp1.expiration, "05/06/28");
        assert_eq!(corp1.name, "Example Brokerage Co");
        assert_eq!(corp1.note, None);

        let corp2 = &record.affiliated_corporations[1];
        assert_eq!(corp2.id, "00000102");
        assert_eq!(corp2.expiration, "08/24/04");
        assert_eq!(corp2.name, "Acme Realty Group Inc");
        assert_eq!(
            corp2.note.as_deref(),
            Some("OFFICER LICENSE EXPIRED AS OF 08/25/04")
        );
    }

    // ── 00000102: Corporation, licensed officers, EXPIRED ──

    #[test]
    fn test_parse_corporation_with_licensed_officers_expired() {
        let html = fixture("corporation_expired_with_officers.html");
        let record = parse_license_page(&html).expect("should parse");

        assert_eq!(record.license_id, "00000102");
        assert_eq!(record.name.as_deref(), Some("Acme Realty Group Inc"));
        assert_eq!(record.license_type.as_deref(), Some("CORPORATION"));
        assert_eq!(
            record.mailing_address.as_deref(),
            Some("100 MAIN ST, SPRINGFIELD, CA 90000")
        );
        assert_eq!(record.expiration_date.as_deref(), Some("08/24/04"));
        assert_eq!(record.status.as_deref(), Some("EXPIRED"));
        assert_eq!(
            record.issued.as_deref(),
            Some("07/08/68 (Unofficial -- taken from secondary records)")
        );
        assert_eq!(record.main_office, None); // "NO CURRENT MAIN OFFICE ADDRESS ON FILE"
        assert_eq!(record.dbas, None);
        assert_eq!(record.branches, None);

        assert_eq!(record.licensed_officers.len(), 2);
        let o1 = &record.licensed_officers[0];
        assert_eq!(o1.id, "00000101");
        assert_eq!(o1.expiration, "08/24/04");
        assert_eq!(o1.name, "Doe, John Quincy");
        assert_eq!(
            o1.note.as_deref(),
            Some("OFFICER LICENSE EXPIRED AS OF 08/25/04")
        );
        let o2 = &record.licensed_officers[1];
        assert_eq!(o2.id, "00000104");
        assert_eq!(o2.name, "Roe, Richard T");
    }

    // ── 00000201: Broker, EXPIRED, no affiliated corps ──

    #[test]
    fn test_parse_broker_expired_no_affiliations() {
        let html = fixture("broker_expired.html");
        let record = parse_license_page(&html).expect("should parse");

        assert_eq!(record.license_id, "00000201");
        assert_eq!(record.name.as_deref(), Some("Roe, William"));
        assert_eq!(record.license_type.as_deref(), Some("BROKER"));
        assert_eq!(record.status.as_deref(), Some("EXPIRED"));
        assert_eq!(record.expiration_date.as_deref(), Some("02/18/12"));
        assert_eq!(
            record.issued.as_deref(),
            Some("02/19/92 (Unofficial -- taken from secondary records)")
        );
        assert_eq!(record.affiliated_corporations.len(), 0);
        assert!(record.disciplinary_comment.is_none());
    }

    // ── 01234567: Salesperson, LICENSED, responsible broker ──

    #[test]
    fn test_parse_salesperson_licensed_with_responsible_broker() {
        let html = fixture("salesperson_active_with_responsible_broker.html");
        let record = parse_license_page(&html).expect("should parse");

        assert_eq!(record.license_id, "01234567");
        assert_eq!(record.name.as_deref(), Some("Doe, John Robert"));
        assert_eq!(record.license_type.as_deref(), Some("SALESPERSON"));
        assert_eq!(record.status.as_deref(), Some("LICENSED"));
        assert_eq!(record.expiration_date.as_deref(), Some("05/07/27"));
        assert_eq!(
            record.issued.as_deref(),
            Some("03/25/83 (Unofficial -- taken from secondary records)")
        );
        assert_eq!(
            record.responsible_broker.as_deref(),
            Some("License ID: 00000103, Example Brokerage Co, 100 MAIN ST, SPRINGFIELD, CA 90000")
        );
        assert_eq!(
            record.former_responsible_broker.as_deref(),
            Some("License ID: 00000103, Example Brokerage Co, From 11/12/2004 to 03/24/2023")
        );
    }

    // ── 09999999: Salesperson, EXPIRED, no responsible broker ──

    #[test]
    fn test_parse_salesperson_expired_no_responsible_broker() {
        let html = fixture("salesperson_expired.html");
        let record = parse_license_page(&html).expect("should parse");

        assert_eq!(record.license_id, "09999999");
        assert_eq!(record.name.as_deref(), Some("White, Patricia F"));
        assert_eq!(record.license_type.as_deref(), Some("SALESPERSON"));
        assert_eq!(record.status.as_deref(), Some("EXPIRED"));
        assert_eq!(record.responsible_broker, None); // "NO CURRENT RESPONSIBLE BROKER"
    }

    // ── Name search: "Doe, John" → 3 rows ──

    #[test]
    fn test_parse_name_search_doe_john_three_matches() {
        let html = fixture("name_search_three_results.html");
        let matches = parse_name_search(&html);

        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0].license_id, "00000105");
        assert_eq!(matches[0].name, "Doe, John Franklin");
        assert_eq!(matches[0].license_type, "Salesperson");
        assert_eq!(matches[0].city, "RIVERTON");

        assert_eq!(matches[1].license_id, "01234567");
        assert_eq!(matches[1].name, "Doe, John Robert");

        assert_eq!(matches[2].license_id, "00000101");
        assert_eq!(matches[2].name, "Doe, John Quincy");
        assert_eq!(matches[2].license_type, "Officer");
        assert_eq!(matches[2].city, "SPRINGFIELD");
    }

    // ── Tool-level argument validation (no network) ──

    #[tokio::test]
    async fn test_dre_lookup_requires_one_of_license_id_or_name() {
        let result = dre_lookup(None, None, None).await;
        assert!(result.is_error);
        assert!(result.content.contains("requires either license_id or name"));
    }

    #[tokio::test]
    async fn test_dre_lookup_rejects_both_license_id_and_name() {
        let result = dre_lookup(Some("00000101"), Some("Doe, John"), None).await;
        assert!(result.is_error);
        assert!(result.content.contains("not both"));
    }
}
