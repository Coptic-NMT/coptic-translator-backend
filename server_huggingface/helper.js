GREEK_TO_COPTIC = {
  α: "ⲁ",
  β: "ⲃ",
  γ: "ⲅ",
  δ: "ⲇ",
  ε: "ⲉ",
  ϛ: "ⲋ",
  ζ: "ⲍ",
  η: "ⲏ",
  θ: "ⲑ",
  ι: "ⲓ",
  κ: "ⲕ",
  λ: "ⲗ",
  μ: "ⲙ",
  ν: "ⲛ",
  ξ: "ⲝ",
  ο: "ⲟ",
  π: "ⲡ",
  ρ: "ⲣ",
  σ: "ⲥ",
  τ: "ⲧ",
  υ: "ⲩ",
  φ: "ⲫ",
  χ: "ⲭ",
  ψ: "ⲯ",
  ω: "ⲱ",
  s: "ϣ",
  f: "ϥ",
  k: "ϧ",
  h: "ϩ",
  j: "ϫ",
  c: "ϭ",
  t: "ϯ",
};

COPTIC_TO_GREEK = {
  ⲁ: "α",
  ⲃ: "β",
  ⲅ: "γ",
  ⲇ: "δ",
  ⲉ: "ε",
  ⲋ: "ϛ",
  ⲍ: "ζ",
  ⲏ: "η",
  ⲑ: "θ",
  ⲓ: "ι",
  ⲕ: "κ",
  ⲗ: "λ",
  ⲙ: "μ",
  ⲛ: "ν",
  ⲝ: "ξ",
  ⲟ: "ο",
  ⲡ: "π",
  ⲣ: "ρ",
  ⲥ: "σ",
  ⲧ: "τ",
  ⲩ: "υ",
  ⲫ: "φ",
  ⲭ: "χ",
  ⲯ: "ψ",
  ⲱ: "ω",
  ϣ: "s",
  ϥ: "f",
  ϧ: "k",
  ϩ: "h",
  ϫ: "j",
  ϭ: "c",
  ϯ: "t",
};

function degreekify(sentence) {
  let newSentence = "";
  for (let i = 0; i < sentence.length; i++) {
    const char = sentence[i];
    if (GREEK_TO_COPTIC[char]) {
      newSentence += GREEK_TO_COPTIC[char];
    } else {
      newSentence += char;
    }
  }
  return newSentence;
}

function greekify(sentence) {
  let newSentence = "";
  for (let i = 0; i < sentence.length; i++) {
    const char = sentence[i];
    if (COPTIC_TO_GREEK[char]) {
      newSentence += COPTIC_TO_GREEK[char];
    } else {
      newSentence += char;
    }
  }
  return newSentence;
}

module.exports = {
    degreekify,
    greekify,
  };