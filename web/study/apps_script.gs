// Response store for the WebVIO user study (web/study, claude/STUDY.md §6). Paste into a Google Apps Script project
// bound to a Google Sheet (Extensions -> Apps Script), then Deploy -> New deployment -> Web app, "Execute as: Me",
// "Who has access: Anyone". Put the web app URL into web/study/common.js STORE_URL.
// POST (text/plain JSON {id, stage, t, ua, data}): appends one row to the sheet "responses".
// GET ?action=order: the next participant's version order (row of common.js ORDERS), from a list shuffled in advance
// (20 blocks, each a permutation of the 6 Latin-square rows; seed 1107), then cycling.
// No IP address or Google account is visible to the script; the page sends no name or e-mail.
const SHEET = 'responses';
const ORDER_LIST = [2,0,4,3,1,5,5,2,1,4,0,3,0,5,2,1,3,4,2,0,5,3,4,1,4,3,1,5,2,0,0,4,2,3,1,5,5,3,2,4,1,0,5,1,0,2,4,3,
                    4,5,3,2,0,1,4,1,3,0,5,2,4,0,3,1,5,2,0,1,2,5,3,4,4,3,0,1,2,5,0,4,1,5,3,2,4,0,1,5,2,3,1,2,3,0,5,4,
                    4,1,2,5,3,0,1,5,0,2,3,4,1,2,0,5,3,4,1,5,4,0,2,3];

function json_(o) {
  return ContentService.createTextOutput(JSON.stringify(o)).setMimeType(ContentService.MimeType.JSON);
}

function doPost(e) {
  const lock = LockService.getScriptLock();
  lock.waitLock(10000);
  try {
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    const sh = ss.getSheetByName(SHEET) || ss.insertSheet(SHEET);
    if (sh.getLastRow() === 0) sh.appendRow(['received', 'participant', 'stage', 'client_time', 'user_agent', 'data_json']);
    const row = JSON.parse(e.postData.contents);
    sh.appendRow([new Date(), String(row.id).slice(0, 32), String(row.stage).slice(0, 32), String(row.t).slice(0, 40),
                  String(row.ua).slice(0, 300), JSON.stringify(row.data).slice(0, 45000)]);
    return json_({ ok: true });
  } catch (err) {
    return json_({ ok: false, error: String(err) });
  } finally {
    lock.releaseLock();
  }
}

function doGet(e) {
  if (e && e.parameter && e.parameter.action === 'order') {
    const lock = LockService.getScriptLock();
    lock.waitLock(10000);
    try {
      const p = PropertiesService.getScriptProperties();
      const n = Number(p.getProperty('next') || 0);
      p.setProperty('next', String(n + 1));
      return json_({ row: ORDER_LIST[n % ORDER_LIST.length], n: n });
    } finally {
      lock.releaseLock();
    }
  }
  return json_({ ok: true });
}
