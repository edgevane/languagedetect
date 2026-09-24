use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use std::collections::HashMap;

use crate::builder::{encode_combined, finish, Engine};
use crate::model::{CombinedDb, LangDb};
use crate::score::classify;

const PL: &[&str] = &[
    "Wczoraj padał deszcz, więc zostałem w domu i czytałem książkę.",
    "Polskie znaki diakrytyczne to między innymi ą, ć, ę, ł, ń, ó, ś, ź oraz ż.",
    "Przyjechałem do Krakowa pociągiem i zwiedzałem Stare Miasto.",
    "Czy mógłbyś mi powiedzieć, która godzina odjeżdża następny autobus?",
    "Wiosną ptaki śpiewają głośno, a drzewa pokrywają się zielenią.",
    "Rząd ogłosił wczoraj nowe przepisy dotyczące ochrony środowiska.",
    "Moja babcia piecze najlepsze ciasto drożdżowe z kruszonką.",
    "Na rynku w Poznaniu kupiłem świeże warzywa i owoce.",
    "Uczę się języka angielskiego już od pięciu lat w szkole.",
    "Zimą w górach pada śnieg i można jeździć na nartach.",
    "Książka leżała na stole obok filiżanki gorącej herbaty.",
    "Dzieci bawiły się wesoło na placu zabaw koło szkoły.",
    "Samolot przyleciał z opóźnieniem z powodu gęstej mgły.",
    "Lekarz zalecił pacjentowi odpoczynek i picie dużej ilości wody.",
    "W muzeum można zobaczyć wiele cennych obrazów i rzeźb.",
];

const EN: &[&str] = &[
    "Yesterday it was raining, so I stayed home and read a book.",
    "The quick brown fox jumps over the lazy dog near the river.",
    "I travelled to London by train and visited the old town.",
    "Could you please tell me when the next bus leaves?",
    "In spring the birds sing loudly and the trees turn green.",
    "The government announced new environmental rules yesterday.",
    "My grandmother bakes the best apple pie with cinnamon.",
    "At the market I bought fresh vegetables and fruit.",
    "I have been learning Spanish for five years at school.",
    "In winter it snows in the mountains and you can ski.",
    "The book lay on the table next to a cup of hot tea.",
    "The children were playing happily in the playground.",
    "The plane arrived late because of the thick fog.",
    "The doctor advised the patient to rest and drink water.",
    "The museum holds many valuable paintings and sculptures.",
];

const DE: &[&str] = &[
    "Gestern hat es geregnet, deshalb bin ich zu Hause geblieben.",
    "Die deutsche Sprache hat interessante Wörter mit Umlauten wie Äpfel.",
    "Ich bin mit dem Zug nach Berlin gefahren und habe Museen besucht.",
    "Könnten Sie mir bitte sagen, wann der nächste Bus abfährt?",
    "Im Frühling singen die Vögel laut und die Bäume werden grün.",
    "Die Regierung hat gestern neue Vorschriften zum Umweltschutz verkündet.",
    "Meine Großmutter backt den besten Apfelstrudel mit Zimt.",
    "Auf dem Markt habe ich frisches Gemüse und Obst gekauft.",
    "Ich lerne seit fünf Jahren Englisch in der Schule.",
    "Im Winter schneit es in den Bergen und man kann Ski fahren.",
    "Das Buch lag auf dem Tisch neben einer Tasse heißem Tee.",
    "Die Kinder haben fröhlich auf dem Spielplatz gespielt.",
    "Das Flugzeug ist wegen dichten Nebels mit Verspätung gelandet.",
    "Der Arzt hat dem Patienten Ruhe und viel Wasser empfohlen.",
    "Im Museum kann man viele wertvolle Gemälde und Skulpturen sehen.",
];

const JA: &[&str] = &[
    "昨日は雨が降っていたので、家にいて本を読んでいました。",
    "日本語の文章にはひらがなとカタカナと漢字が混ざっています。",
    "電車で京都に行って、古いお寺を見学しました。",
    "次のバスは何時に出発するか教えていただけますか。",
    "春になると鳥がさえずり、木々が緑に覆われます。",
    "政府は昨日新しい環境規制を発表しました。",
    "祖母が作るアップルパイはとても美味しいです。",
    "市場で新鮮な野菜と果物を買いました。",
    "学校で五年以上英語を勉強しています。",
    "冬には山で雪が降り、スキーができます。",
    "本は温かいお茶の横のテーブルの上にありました。",
    "子供たちは校庭で元気に遊んでいました。",
    "濃い霧のため飛行機が遅れて到着しました。",
    "医者は患者に休息と水分補給を勧めました。",
    "博物館では多くの貴重な絵画や彫刻が見られます。",
];

fn train(docs: &[&str]) -> Engine {
    let mut e = Engine::new();
    for d in docs {

        for _ in 0..20 {
            e.ingest(d);
        }
    }
    e
}


fn toy_idf(engines: &[Engine]) -> BTreeMap<u32, f32> {
    let n = engines.len() as f32;
    let mut df: HashMap<u32, u32> = HashMap::new();
    for e in engines {
        for (&tok, _) in e.token_df() {
            *df.entry(tok).or_insert(0) += 1;
        }
    }
    df.into_iter()
        .map(|(tok, d)| {
            (
                tok,
                libm::logf((n - d as f32 + 0.5) / (d as f32 + 0.5) + 1.0),
            )
        })
        .collect()
}

fn build_all() -> Vec<(String, Vec<u8>)> {
    let engines = [train(PL), train(EN), train(DE), train(JA)];
    let idf = toy_idf(&engines);
    ["pl", "en", "de", "ja"]
        .into_iter()
        .zip(engines.iter())
        .map(|(code, e)| (String::from(code), finish(e, code, false, &idf)))
        .collect()
}

#[test]
fn blobs_fit_budget_and_parse() {
    for (code, blob) in build_all() {
        assert!(blob.len() < 300 * 1024, "{code}: {} bytes", blob.len());
        let db = LangDb::from_bytes(&blob).expect("parse");
        assert_eq!(db.lang_code(), code);
    }
}

#[test]
fn determinism() {
    let a = build_all();
    let b = build_all();
    for ((ca, ba), (cb, bb)) in a.iter().zip(b.iter()) {
        assert_eq!(ca, cb);
        assert_eq!(ba, bb, "non-deterministic blob for {ca}");
    }
}

#[test]
fn classify_heldout() {
    let items = build_all();
    let dbs: Vec<LangDb> = items
        .iter()
        .map(|(_, b)| LangDb::from_bytes(b).unwrap())
        .collect();
    let cases = [
        ("W Krakowie pada śnieg i jest bardzo zimno w styczniu.", "pl"),
        ("The weather in Scotland is often cold and rainy.", "en"),
        ("Die Kinder spielen im Garten mit dem neuen Hund.", "de"),
        ("明日は晴れるので公園に散歩に行きます。", "ja"),
    ];
    for (text, want) in cases {
        let ranked = classify(&dbs, text);
        assert_eq!(ranked[0].lang_code(), want, "misclassified: {text}");
        assert!(
            ranked[0].confidence > 0.5,
            "low confidence for {want}: {}",
            ranked[0].confidence
        );
    }
}

#[test]
fn classify_short_snippets() {
    let items = build_all();
    let dbs: Vec<LangDb> = items
        .iter()
        .map(|(_, b)| LangDb::from_bytes(b).unwrap())
        .collect();
    let cases = [
        ("Dzień dobry, poproszę kawę.", "pl"),
        ("Good morning, a coffee please.", "en"),
        ("Guten Morgen, einen Kaffee bitte.", "de"),
        ("おはようございます。", "ja"),
    ];
    for (text, want) in cases {
        let ranked = classify(&dbs, text);
        assert_eq!(ranked[0].lang_code(), want, "misclassified: {text}");
    }
}

#[test]
fn combined_roundtrip_and_find() {
    let items = build_all();
    let bundle = encode_combined(&items);
    let cdb = CombinedDb::from_bytes(&bundle).expect("parse combined");
    assert_eq!(cdb.len(), 4);
    let de = cdb.find("de").expect("find de");
    assert_eq!(de.lang_code(), "de");
    assert!(cdb.find("xx").is_none());
    let single: Vec<LangDb> = items
        .iter()
        .map(|(_, b)| LangDb::from_bytes(b).unwrap())
        .collect();
    let from_bundle: Vec<LangDb> = (0..cdb.len()).map(|i| cdb.get(i).unwrap()).collect();
    let text = "Czytanie książek to wspaniała przygoda dla każdego dziecka.";
    let a = classify(&single, text);
    let b = classify(&from_bundle, text);
    assert_eq!(a[0].lang_code(), b[0].lang_code());
}

#[test]
fn corruption_is_rejected() {
    let items = build_all();
    let mut blob = items[0].1.clone();
    blob[100] ^= 0xFF;
    assert!(LangDb::from_bytes(&blob).is_err());
}
