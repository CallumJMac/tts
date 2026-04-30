/**
 * Google Apps Script for collecting MUSHRA listening study results.
 *
 * Deploy as a Web App:
 *   1. Paste this into Extensions > Apps Script in a Google Sheet
 *   2. Deploy > New deployment > Web app
 *   3. Execute as: Me | Access: Anyone
 *   4. Copy the URL into configs/study.json "resultsEndpoint"
 */

function doPost(e) {
  try {
    const data = JSON.parse(e.postData.contents);
    const ss = SpreadsheetApp.getActiveSpreadsheet();

    // Raw JSON sheet (one row per submission)
    let rawSheet = ss.getSheetByName("Raw") || ss.insertSheet("Raw");
    if (rawSheet.getLastRow() === 0) {
      rawSheet.appendRow(["timestamp", "participant_id", "duration_ms", "user_agent", "json"]);
    }
    rawSheet.appendRow([
      new Date().toISOString(),
      data.participantId,
      data.durationMs,
      data.userAgent,
      JSON.stringify(data)
    ]);

    // Ratings sheet (one row per trial per participant)
    let ratingsSheet = ss.getSheetByName("Ratings") || ss.insertSheet("Ratings");
    if (ratingsSheet.getLastRow() === 0) {
      ratingsSheet.appendRow([
        "timestamp", "participant_id", "trial_id",
        "condition", "naturalness", "similarity"
      ]);
    }

    const mushraResults = data.results.filter(r => r.type === "mushra");
    for (const trial of mushraResults) {
      for (const [cond, scores] of Object.entries(trial.ratings)) {
        ratingsSheet.appendRow([
          new Date().toISOString(),
          data.participantId,
          trial.trialId,
          cond,
          scores.naturalness,
          scores.similarity
        ]);
      }
    }

    return ContentService.createTextOutput(
      JSON.stringify({ status: "ok" })
    ).setMimeType(ContentService.MimeType.JSON);

  } catch (err) {
    return ContentService.createTextOutput(
      JSON.stringify({ status: "error", message: err.toString() })
    ).setMimeType(ContentService.MimeType.JSON);
  }
}

function doGet(e) {
  return ContentService.createTextOutput("MUSHRA results endpoint active.");
}
