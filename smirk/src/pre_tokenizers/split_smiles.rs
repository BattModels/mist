use const_format::formatcp;

const BRACKETED_SYMBOL: &'static str = concat!(
    r"A[c|g|l|m|r|s|t|u]|",
    r"B[a|e|h|i|k|r]?|",
    r"C[a|d|e|f|l|m|n|o|r|s|u]?|",
    r"D[b|s|y]|",
    r"E[r|s|u]|",
    r"F[e|l|m|r]?|",
    r"G[a|d|e]|",
    r"H[e|f|g|o|s]?|",
    r"I[n|r]?|",
    r"Kr?|",
    r"L[a|i|r|u|v]|",
    r"M[c|d|g|n|o|t]|",
    r"N[a|b|d|e|h|i|o|p]?|",
    r"O[g|s]?|",
    r"P[a|b|d|m|o|r|t|u]?|",
    r"R[a|b|e|f|g|h|n|u]|",
    r"S[b|c|e|g|i|m|n|r]?|",
    r"T[a|b|c|e|h|i|l|m|s]|",
    r"U|",
    r"V|",
    r"W|",
    r"Xe|",
    r"Yb?|",
    r"Z[n|r]|",
    r"as|",
    r"b|",
    r"c|",
    r"n|",
    r"o|",
    r"p|",
    r"se?|",
    r"\*",
);

const CHIRAL: &'static str = r"@(?:@|AL|OH|SP|T[B|H])?";

pub const MATCH_OUTER: &'static str = concat!(
    r"Br?|Cl?|F|I|N|O|P|S|",
    r"b|c|n|o|p|s|",
    r"\*|",
    r"[\.\-=\#\$:/\\]|",
    r"\d|%|",
    r"\(|\)|",
    r"\[.*?]",
);

pub const MATCH_INNER: &'static str = formatcp!(concat!(
    r"(\d+)?",                        // Isotope
    r"({BRACKETED_SYMBOL})",          // Element Symbols
    r"(?:({CHIRAL})(\d{{1,2}})?)?",   // Chirality
    r"(?:(H)(\d)?)?",                 // Hydrogen Count
    r"(?:([+-]{{1,2}})(\d{{0,2}}))?", // Charge
    r"(?:(:)(\d+))?",                 // Class
));
