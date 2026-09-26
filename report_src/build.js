// IEEE conference-style report for the multimodal product recommendation project.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, WidthType,
  BorderStyle, ShadingType, SectionType, TabStopType, LevelFormat, VerticalAlign,
} = require("docx");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const ROOT = "D:/D drive/Project/claude/multimodal_recsys";
const FONT = "Times New Roman";
const COL = 5040;          // one IEEE column: 3.5 in
const FULL = 10440;        // full text width: 7.25 in
const f3 = (x) => (x === null || x === undefined || Number.isNaN(x) ? "—" : Number(x).toFixed(3));
const ms = (r, k) => (r[`${k} mean`] === null || Number.isNaN(r[`${k} mean`]) ? "—" : `${f3(r[`${k} mean`])} ± ${f3(r[`${k} std`])}`);
const byKey = (arr, key, val) => arr.find((r) => r[key] === val);

// ---------------------------------------------------------------- text helpers
// inline markup: **bold**, *italic*, _{sub}, ^{sup}
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (t.startsWith("*")) out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    else if (t.startsWith("_{")) out.push(new TextRun({ text: t.slice(2, -1), subScript: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: t.slice(2, -1), superScript: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}
const SZ = 20; // 10 pt body
const P = (text, opt = {}) => new Paragraph({
  children: runs(text, { size: opt.size || SZ }), alignment: opt.align || AlignmentType.JUSTIFIED,
  indent: opt.noIndent ? undefined : { firstLine: 202 }, spacing: { after: opt.after ?? 0, line: 228 },
});
const H1 = (num, text) => new Paragraph({
  children: [new TextRun({ text: `${num}. ${text}`, font: FONT, size: SZ, smallCaps: true })],
  alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 }, keepNext: true,
});
const H2 = (letter, text) => new Paragraph({
  children: [new TextRun({ text: `${letter}. ${text}`, font: FONT, size: SZ, italics: true })],
  spacing: { before: 120, after: 60 }, keepNext: true,
});
const EQ = (text, num, width = COL) => new Paragraph({
  children: [new TextRun({ text: "\t", font: FONT }), ...runs(text, { size: SZ, italics: false }),
             new TextRun({ text: `\t(${num})`, font: FONT, size: SZ })],
  tabStops: [{ type: TabStopType.CENTER, position: width / 2 }, { type: TabStopType.RIGHT, position: width }],
  spacing: { before: 80, after: 80 },
});
const BULLET = (text) => new Paragraph({
  children: runs(text, { size: SZ }), numbering: { reference: "bullets", level: 0 },
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 20, line: 228 },
});
const TCAP = (roman, title) => [
  new Paragraph({ children: [new TextRun({ text: `TABLE ${roman}`, font: FONT, size: 16, smallCaps: true })],
                  alignment: AlignmentType.CENTER, spacing: { before: 160, after: 0 }, keepNext: true }),
  new Paragraph({ children: [new TextRun({ text: title, font: FONT, size: 16, smallCaps: true })],
                  alignment: AlignmentType.CENTER, spacing: { after: 60 }, keepNext: true }),
];
const FIG = (file, widthIn, n, caption) => {
  const buf = fs.readFileSync(file);
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20); // PNG header
  const pxW = Math.round(widthIn * 96);
  return [
    new Paragraph({ children: [new ImageRun({ type: "png", data: buf, transformation: { width: pxW, height: Math.round(pxW * h / w) } })],
                    alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true }),
    new Paragraph({ children: runs(`Fig. ${n}. ${caption}`, { size: 16 }), alignment: AlignmentType.JUSTIFIED,
                    spacing: { after: 140 } }),
  ];
};

// ---------------------------------------------------------------- table helper
const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function TABLE(headers, rows, widths, opt = {}) {
  const size = opt.size || 15;
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, w, isHead, bold, shade) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    margins: { top: 30, bottom: 30, left: 60, right: 60 },
    verticalAlign: VerticalAlign.CENTER,
    borders: { top: isHead ? border : noBorder, bottom: isHead ? border : noBorder, left: noBorder, right: noBorder },
    shading: shade ? { fill: "E8F3EC", type: ShadingType.CLEAR, color: "auto" } : undefined,
    children: [new Paragraph({ children: runs(String(text), { size, bold: isHead || bold }),
                               alignment: opt.leftCols && opt.leftCols.includes(w) ? AlignmentType.LEFT : AlignmentType.CENTER })],
  });
  const head = new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, widths[i], true)) });
  const body = rows.map((r, ri) => new TableRow({
    cantSplit: true,
    children: r.cells.map((c, i) => {
      const tc = cell(c, widths[i], false, r.bold, r.shade);
      return tc;
    }),
  }));
  // bottom rule on the last row
  const lastRow = body[body.length - 1];
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: [head, ...body],
    borders: { top: border, bottom: border, left: noBorder, right: noBorder, insideHorizontal: noBorder, insideVertical: noBorder },
  });
}
// left-aligned text columns: pass leftIdx to align specific columns left
function TABLE2(headers, rows, widths, leftIdx = [0], size = 15, keep = true) {
  const total = widths.reduce((a, b) => a + b, 0);
  const mk = (text, i, head, bold, shade) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, margins: { top: 25, bottom: 25, left: 50, right: 50 },
    verticalAlign: VerticalAlign.CENTER,
    borders: { top: head ? border : noBorder, bottom: head ? border : noBorder, left: noBorder, right: noBorder },
    shading: shade ? { fill: "E8F3EC", type: ShadingType.CLEAR, color: "auto" } : undefined,
    children: [new Paragraph({ children: runs(String(text), { size, bold: head || bold }), keepNext: keep,
                               alignment: leftIdx.includes(i) ? AlignmentType.LEFT : AlignmentType.CENTER })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    borders: { top: border, bottom: border, left: noBorder, right: noBorder, insideHorizontal: noBorder, insideVertical: noBorder },
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => mk(h, i, true)) }),
           ...rows.map((r) => new TableRow({ cantSplit: true, children: r.cells.map((c, i) => mk(c, i, false, r.bold, r.shade)) }))],
  });
}

// ---------------------------------------------------------------- data
const S = D.summary;
const M = Object.fromEntries(D.methods.map((r) => [r.method, r]));
const B = Object.fromEntries(D.balance.map((r) => [r.variant, r]));
const O = Object.fromEntries(D.optimizers.map((r) => [r.optimizer, r]));
const A = Object.fromEntries(D.ablations.map((r) => [r.ablation, r]));
const rob = (model, cond, k = "i2i") => { const r = D.robust.find((x) => x.model === model && x.condition === cond); return r ? r[`${k} mean`] : NaN; };
const psig = (variant, metric) => byKey(D.balance_sig.filter((r) => r.variant === variant), "metric", metric);
const pstr = (p) => (p < 0.001 ? "p < 0.001" : `p = ${p.toFixed(3)}`);
const G = B.uni;                      // final proposed model
const K = ["i2i", "i2i_strict", "text2item", "image2item"];
const nTest = S.n_test, nProd = S.n_products;

// ---------------------------------------------------------------- literature table rows (verified; see LITERATURE_REVIEW.md)
const LIT = [
  [1, "LATTICE: Mining latent structures for multimedia recommendation", "J. Zhang et al.", "2021, ACM MM", "Modality-aware item–item graph learning + graph convolution", "Beats SOTA multimedia recommenders on 3 datasets", "Needs interaction data; frozen pretrained features"],
  [2, "MICRO: Latent structure mining with contrastive modality fusion", "J. Zhang et al.", "2023, IEEE TKDE", "LATTICE + contrastive shared/specific fusion", "Improves LATTICE-style baselines", "Interaction-dependent; no single-modality queries"],
  [3, "MMGCL: Multi-modal graph contrastive learning for micro-video rec.", "Z. Yi et al.", "2022, SIGIR", "Graph contrastive learning with modality masking", "Better accuracy and faster convergence on 2 datasets", "Masking only as augmentation, not missing queries"],
  [4, "BM3: Bootstrap latent representations for multi-modal rec.", "X. Zhou et al.", "2023, WWW", "Bootstrapped self-supervision, cross-modal alignment", "Beats prior models; 2–9× less training time", "Pre-extracted features; ID embeddings"],
  [5, "MMSSL: Multi-modal self-supervised learning for rec.", "W. Wei et al.", "2023, WWW", "Adversarial structure learning + cross-modal contrast", "Outperforms SOTA multimodal recommenders", "Complex training; not end-to-end"],
  [6, "FREEDOM: Freezing and denoising graph structures", "X. Zhou, Z. Shen", "2023, ACM MM", "Frozen item graph + denoised user–item graph", "Competitive accuracy at much lower cost", "Fixed pretrained features; graph-level fusion"],
  [7, "DRAGON: Homogeneous graphs for multimodal rec.", "H. Zhou et al.", "2023, ECAI", "User–user and item–item graphs, attentive concatenation", "Shows common fusions can hurt accuracy", "Concatenation fusion; interaction-dependent"],
  [8, "LGMRec: Local and global graph learning", "Z. Guo et al.", "2024, AAAI", "Local graph + global hypergraph learning", "Improves graph-based recommenders", "Heavy model; no missing-modality design"],
  [9, "MENTOR: Multi-level self-supervised learning", "J. Xu et al.", "2025, AAAI", "ID-guided multi-level cross-modal alignment", "Reduces the modality gap", "Needs interaction history"],
  [10, "Modality-balanced learning for multimedia rec. (CKD)", "J. Zhang et al.", "2024, ACM MM", "Counterfactual knowledge distillation", "Text-only can beat multimodal; CKD fixes it", "Pre-extracted features; extra distillation"],
  [11, "Where to go next? ID- vs. modality-based recommenders", "Z. Yuan et al.", "2023, SIGIR", "End-to-end trained modality encoders vs IDs", "Modality-based models can rival ID-based", "Large pretrained encoders; no fusion study"],
  [12, "Do we need to drop items with missing modalities?", "D. Malitesta et al.", "2024, CIKM", "Graph-based imputation of missing features", "Dropping items is unnecessary and harmful", "Imputes before training, not at query time"],
  [13, "Does multimodality improve recommenders as expected?", "H. Zhou et al.", "2025, arXiv", "Large re-evaluation study", "Gains modest; text matters most in e-commerce; 29.3% reproducible", "Analysis only, no new model"],
  [14, "Multimodal recommender systems: A survey", "Q. Liu et al.", "2024, ACM CSUR", "Survey", "Taxonomy of multimodal recommenders", "Little on from-scratch encoders"],
  [15, "Multimodal pretraining, adaptation and generation for rec.: A survey", "Q. Liu et al.", "2024, KDD", "Survey", "Maps the move to large pretrained models", "Low-resource training under-explored"],
  [16, "CLIP: Learning transferable visual models from NL supervision", "A. Radford et al.", "2021, ICML", "Image–text dual encoder, 400M pairs", "Zero-shot transfer on 30+ datasets", "Web-scale data and compute"],
  [17, "SigLIP: Sigmoid loss for language–image pre-training", "X. Zhai et al.", "2023, ICCV", "Pairwise sigmoid contrastive loss", "84.5% ImageNet zero-shot with 4 TPUs", "Still large-scale pretraining"],
  [18, "FashionCLIP: General fashion concepts", "P. J. Chia et al.", "2022, Sci. Rep.", "CLIP fine-tuned on fashion", "Transferable fashion representations", "Relies on pretrained CLIP; no fusion"],
  [19, "FashionViL: Fashion-focused V+L representation learning", "X. Han et al.", "2022, ECCV", "Fashion V+L Transformer, multi-view contrast", "SOTA on 5 fashion tasks", "Heavy pretraining, large model"],
  [20, "CLIP4Cir: Conditioned and composed image retrieval", "A. Baldrati et al.", "2022, CVPR", "Combiner network over CLIP features", "Composed image + text retrieval", "Pretrained CLIP backbone"],
  [21, "Are multimodal Transformers robust to missing modality?", "M. Ma et al.", "2022, CVPR", "Study + fusion-strategy search", "Sensitive to missing modality; fusion is data-dependent", "Classification, not retrieval"],
  [22, "Multimodal prompting with missing modalities", "Y.-L. Lee et al.", "2023, CVPR", "Missing-aware prompts on ViLT", "Robust with <1% trainable parameters", "Needs a large pretrained backbone"],
  [23, "ShaSpec: Missing modality via shared-specific features", "H. Wang et al.", "2023, CVPR", "Shared + specific features, residual fusion", "+3–5% over SOTA (medical)", "Medical / classification only"],
  [24, "OGM-GE: On-the-fly gradient modulation", "X. Peng et al.", "2022, CVPR", "Per-modality gradient modulation", "Clear gains over standard fusion", "Audio-visual classification only"],
  [25, "On uni-modal feature learning in supervised multi-modal learning", "C. Du et al.", "2023, ICML", "Uni-modal teacher / ensemble", "Identifies modality laziness", "Classification only"],
];

// ---------------------------------------------------------------- references (IEEE style)
const REFS = [
  "J. Zhang, Y. Zhu, Q. Liu, S. Wu, S. Wang, and L. Wang, “Mining latent structures for multimedia recommendation,” in *Proc. 29th ACM Int. Conf. Multimedia (MM)*, 2021.",
  "J. Zhang *et al.*, “Latent structure mining with contrastive modality fusion for multimedia recommendation,” *IEEE Trans. Knowl. Data Eng.*, vol. 35, no. 9, pp. 9154–9167, 2023.",
  "Z. Yi, X. Wang, I. Ounis, and C. Macdonald, “Multi-modal graph contrastive learning for micro-video recommendation,” in *Proc. 45th Int. ACM SIGIR Conf.*, 2022.",
  "X. Zhou *et al.*, “Bootstrap latent representations for multi-modal recommendation,” in *Proc. ACM Web Conf. (WWW)*, 2023, pp. 845–854.",
  "W. Wei, C. Huang, L. Xia, and C. Zhang, “Multi-modal self-supervised learning for recommendation,” in *Proc. ACM Web Conf. (WWW)*, 2023, pp. 790–800.",
  "X. Zhou and Z. Shen, “A tale of two graphs: Freezing and denoising graph structures for multimodal recommendation,” in *Proc. 31st ACM Int. Conf. Multimedia (MM)*, 2023.",
  "H. Zhou, X. Zhou, and Z. Shen, “Enhancing dyadic relations with homogeneous graphs for multimodal recommendation,” in *Proc. 26th Eur. Conf. Artif. Intell. (ECAI)*, 2023.",
  "Z. Guo *et al.*, “LGMRec: Local and global graph learning for multimodal recommendation,” in *Proc. AAAI Conf. Artif. Intell.*, vol. 38, no. 8, 2024, pp. 8454–8462.",
  "J. Xu, Z. Chen, S. Yang, J. Li, H. Wang, and E. C. H. Ngai, “MENTOR: Multi-level self-supervised learning for multimodal recommendation,” in *Proc. AAAI Conf. Artif. Intell.*, vol. 39, no. 12, 2025, pp. 12908–12917.",
  "J. Zhang, G. Liu, Q. Liu, S. Wu, and L. Wang, “Modality-balanced learning for multimedia recommendation,” in *Proc. 32nd ACM Int. Conf. Multimedia (MM)*, 2024, pp. 7551–7560.",
  "Z. Yuan *et al.*, “Where to go next for recommender systems? ID- vs. modality-based recommender models revisited,” in *Proc. 46th Int. ACM SIGIR Conf.*, 2023.",
  "D. Malitesta, E. Rossi, C. Pomo, T. Di Noia, and F. D. Malliaros, “Do we really need to drop items with missing modalities in multimodal recommendation?” in *Proc. 33rd ACM Int. Conf. Inf. Knowl. Manage. (CIKM)*, 2024.",
  "H. Zhou, Y. Zhang, A. Sun, and Z. Shen, “Does multimodality improve recommender systems as expected? A critical analysis and future directions,” arXiv:2508.05377, 2025.",
  "Q. Liu *et al.*, “Multimodal recommender systems: A survey,” *ACM Comput. Surv.*, vol. 57, no. 2, 2024.",
  "Q. Liu *et al.*, “Multimodal pretraining, adaptation, and generation for recommendation: A survey,” in *Proc. 30th ACM SIGKDD Conf. Knowl. Discov. Data Mining (KDD)*, 2024.",
  "A. Radford *et al.*, “Learning transferable visual models from natural language supervision,” in *Proc. 38th Int. Conf. Mach. Learn. (ICML)*, 2021, pp. 8748–8763.",
  "X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2023, pp. 11975–11986.",
  "P. J. Chia *et al.*, “Contrastive language and vision learning of general fashion concepts,” *Sci. Rep.*, vol. 12, Art. no. 18958, 2022.",
  "X. Han, L. Yu, X. Zhu, L. Zhang, Y.-Z. Song, and T. Xiang, “FashionViL: Fashion-focused vision-and-language representation learning,” in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2022.",
  "A. Baldrati, M. Bertini, T. Uricchio, and A. Del Bimbo, “Effective conditioned and composed image retrieval combining CLIP-based features,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022.",
  "M. Ma, J. Ren, L. Zhao, D. Testuggine, and X. Peng, “Are multimodal Transformers robust to missing modality?” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 18177–18186.",
  "Y.-L. Lee, Y.-H. Tsai, W.-C. Chiu, and C.-Y. Lee, “Multimodal prompting with missing modalities for visual recognition,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2023, pp. 14943–14952.",
  "H. Wang, Y. Chen, C. Ma, J. Avery, L. Hull, and G. Carneiro, “Multi-modal learning with missing modality via shared-specific feature modelling,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2023.",
  "X. Peng, Y. Wei, A. Deng, D. Wang, and D. Hu, “Balanced multimodal learning via on-the-fly gradient modulation,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022.",
  "C. Du *et al.*, “On uni-modal feature learning in supervised multi-modal learning,” in *Proc. 40th Int. Conf. Mach. Learn. (ICML)*, 2023.",
  "P. Khosla *et al.*, “Supervised contrastive learning,” in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2020.",
  "J. Arevalo, T. Solorio, M. Montes-y-Gómez, and F. A. González, “Gated multimodal units for information fusion,” in *Proc. Int. Conf. Learn. Represent. (ICLR) Workshop*, 2017.",
  "K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016.",
  "A. Vaswani *et al.*, “Attention is all you need,” in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017.",
  "A. Radford, L. Metz, and S. Chintala, “Unsupervised representation learning with deep convolutional generative adversarial networks,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2016.",
  "S. Zhao, Z. Liu, J. Lin, J.-Y. Zhu, and S. Han, “Differentiable augmentation for data-efficient GAN training,” in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2020.",
  "K. Järvelin and J. Kekäläinen, “Cumulated gain-based evaluation of IR techniques,” *ACM Trans. Inf. Syst.*, vol. 20, no. 4, pp. 422–446, 2002.",
  "P. Aggarwal, “Fashion product images (small),” Kaggle dataset, 2019. [Online]. Available: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small",
  "A. Paszke *et al.*, “PyTorch: An imperative style, high-performance deep learning library,” in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019.",
  "B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York, NY, USA: Chapman & Hall, 1993.",
];

// ---------------------------------------------------------------- content
const title = [
  new Paragraph({ children: [new TextRun({ text: "Gated Multimodal Fusion with Uni-Modal Supervision for Product Recommendation Using Image and Text Embeddings Learned from Scratch", font: FONT, size: 48 })],
                  alignment: AlignmentType.CENTER, spacing: { after: 240 } }),
];
const authorBlock = (lines) => new TableCell({
  width: { size: FULL / 2, type: WidthType.DXA },
  borders: { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder },
  children: lines.map((l, i) => new Paragraph({ children: [new TextRun({ text: l, font: FONT, size: i === 0 ? 22 : 20, italics: i === 1 || i === 2 })],
                                                alignment: AlignmentType.CENTER })),
});
const authors = new Table({
  width: { size: FULL, type: WidthType.DXA }, columnWidths: [FULL / 2, FULL / 2],
  borders: { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder, insideHorizontal: noBorder, insideVertical: noBorder },
  rows: [new TableRow({ children: [
    authorBlock(["[Student Name]", "Dept. of [Department]", "[College / University Name]", "[City], India", "[email address]"]),
    authorBlock(["[Guide Name]", "Dept. of [Department]", "[College / University Name]", "[City], India", "[email address]"]),
  ] })],
});

const pctUp = (a, b) => (100 * (a - b) / b).toFixed(1);
const col1 = [];
// Abstract
col1.push(new Paragraph({
  children: [new TextRun({ text: "Abstract", font: FONT, size: 18, bold: true, italics: true }),
             new TextRun({ text: "—", font: FONT, size: 18, bold: true }),
             ...runs(`Product recommendation benefits from both what a product looks like and how it is described, yet most multimodal recommenders rely on large pretrained encoders and user–item interaction graphs. We study multimodal product recommendation in which both the image encoder and the text encoder are trained entirely from scratch on the product catalog. A ResNet-style CNN embeds product photos, a Transformer with its own tokenizer embeds product titles, and a gated fusion unit learns, per feature and per product, how much to trust each modality. Training combines a supervised contrastive loss, an image–text contrastive loss and a uni-modal supervision term that counters modality imbalance. On ${nProd.toLocaleString("en-US")} fashion products, evaluated on ${nTest.toLocaleString("en-US")} unseen test products with three seeds and paired bootstrap tests, the final model reaches NDCG@10 of ${f3(G["i2i mean"])} for similar-product recommendation, ${f3(G["i2i_strict mean"])} when colour must also match, ${f3(G["text2item mean"])} for text search and ${f3(G["image2item mean"])} for photo search, improving significantly on all four tasks over plain gated fusion. We further compare SGD, Adam and AdamW, run ablations, and introduce a noisy-title benchmark: when titles are missing, a text-only model collapses to ${f3(rob("text", "no title"))} while the fused model retains ${f3(rob("gated+uni", "no title"))}, and noise-aware training keeps accuracy above an image-only model under every title corruption tested.`, { size: 18, bold: true })],
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 },
}));
col1.push(new Paragraph({
  children: [new TextRun({ text: "Index Terms", font: FONT, size: 18, bold: true, italics: true }),
             new TextRun({ text: "—multimodal recommendation, gated fusion, contrastive learning, modality imbalance, missing modality, product retrieval.", font: FONT, size: 18, bold: true })],
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 },
}));

// I. INTRODUCTION
col1.push(H1("I", "Introduction"));
col1.push(P("Online shops recommend products through “similar items” lists and search boxes. A product is described by two complementary signals: its photo, which captures shape, pattern and style, and its title, which names the product type, brand and colour. Multimodal recommendation combines both, and has become an active research area [1]–[15]."));
col1.push(P("Most recent multimodal recommenders, however, feed features from large pretrained networks into graphs built from user–item interactions [1]–[9]. Such systems need interaction histories, which are unavailable for new products (cold start), and depend on pretraining at web scale [16], [17]. Moreover, recent analyses report that text often dominates the image in e-commerce and that multimodal models can even underperform text-only ones [10], [13], a symptom of *modality imbalance* [24], [25]. Finally, real catalogs contain short, missing or wrong titles, a situation that fusion models are rarely tested on [12], [21]."));
col1.push(P("This work studies content-based multimodal product recommendation with encoders trained entirely from scratch, and makes the following contributions:"));
col1.push(BULLET("A lightweight model (3.8 M parameters) with a CNN image encoder, a Transformer title encoder and a **gated fusion** unit [27], trained end-to-end from random initialisation, that answers four query types with one embedding: similar product, text, photo, and photo + text."));
col1.push(BULLET("**Uni-modal supervision** that counters modality imbalance and significantly improves all four tasks; it is compared with gradient modulation (OGM) [24] and noise-aware training."));
col1.push(BULLET("A controlled study of **six fusion strategies** and **three optimizers** (SGD, Adam, AdamW), each tuned on validation, with three seeds, ablations and paired bootstrap significance tests."));
col1.push(BULLET("A **noisy-title benchmark** showing that fusion keeps working when titles are missing, that plain fusion over-trusts damaged text, and that noise-aware training removes this weakness."));

// II. RELATED WORK
col1.push(H1("II", "Related Work"));
col1.push(H2("A", "Multimodal Recommender Systems"));
col1.push(P("Graph-based models dominate recent multimodal recommendation. LATTICE [1] learns modality-aware item–item graphs, and MICRO [2] adds contrastive modality fusion. MMGCL [3], BM3 [4] and MMSSL [5] use self-supervised and contrastive objectives, while FREEDOM [6], DRAGON [7], LGMRec [8] and MENTOR [9] refine graph construction and cross-modal alignment. These methods consume pre-extracted image and text features and require user–item interactions. Yuan *et al.* [11] show that modality encoders trained end-to-end can rival ID-based recommenders, and surveys [14], [15] note the shift towards large pretrained models."));
col1.push(H2("B", "Vision–Language Models and Fashion Retrieval"));
col1.push(P("CLIP [16] and SigLIP [17] learn aligned image and text encoders from hundreds of millions of pairs. FashionCLIP [18] and FashionViL [19] adapt this to fashion, and CLIP4Cir [20] composes a reference image with a text modification. All rely on large pretrained backbones; in contrast, our encoders are learned from the catalog alone."));
col1.push(H2("C", "Missing Modalities and Modality Imbalance"));
col1.push(P("Ma *et al.* [21] show that multimodal Transformers are sensitive to missing modalities; missing-aware prompts [22] and shared–specific features [23] improve robustness in classification, and Malitesta *et al.* [12] impute missing features before training a recommender. Multimodal networks also tend to rely on the easiest modality: OGM-GE [24] modulates per-modality gradients, Du *et al.* [25] identify “modality laziness”, and Zhang *et al.* [10] report that text-only recommenders can beat multimodal ones. Table I summarises the 25 most relevant studies since 2021."));

// full-width literature table section
const litSection = [
  ...TCAP("I", "Summary of Related Work (2021–2025)"),
  TABLE2(["Ref.", "Title (short)", "Authors", "Year, venue", "Model used", "Results reported", "Limitation w.r.t. this work"],
         LIT.map((r) => ({ cells: [`[${r[0]}]`, r[1], r[2], r[3], r[4], r[5], r[6]] })),
         [420, 2350, 1050, 950, 1950, 1950, 1770], [1, 2, 4, 5, 6], 13, false),
  new Paragraph({ children: [], spacing: { after: 80 } }),
];

const col2 = [];
col2.push(P("Compared with these studies, our work (i) trains both encoders from scratch without interaction data, (ii) handles a missing modality *at query time* rather than by imputation, (iii) evaluates fusion under corrupted titles, and (iv) combines gated fusion with uni-modal supervision to counter modality imbalance in product retrieval."));

// III. PROPOSED METHOD
col2.push(H1("III", "Proposed Method"));
col2.push(H2("A", "Problem Formulation"));
col2.push(P("Each product *i* has a photo *x*_{i} and a title *t*_{i}. The goal is an encoder that maps (*x*_{i}, *t*_{i}) to an L2-normalised embedding **f**_{i} such that products of the same type (and colour) are close. Recommendation ranks catalog items by cosine similarity to a query embedding; a query may contain a photo, a title-like text, or both."));
col2.push(H2("B", "Encoders Trained from Scratch"));
col2.push(P("The image encoder is a ResNet-style CNN [28] with a 3×3 stem and six residual blocks (32→64→128→256 channels, three stride-2 stages). A 64×64 photo yields an 8×8×256 feature map, average-pooled into **h**_{img} ∈ ℝ^{256}. Titles are lower-cased and split into words; the vocabulary (3,862 words) is built from training titles only. A 2-layer Transformer [29] (width 128, 4 heads, learned positions, 16 tokens) with masked mean pooling and a linear layer gives **h**_{txt} ∈ ℝ^{256}. No pretrained weights are used anywhere."));
col2.push(H2("C", "Gated Fusion"));
col2.push(P("Following gated multimodal units [27], a gate decides per dimension how much each modality contributes:"));
col2.push(EQ("**g** = σ(**W**_{g}[**h**_{img} ; **h**_{txt}] + **b**_{g})", 1));
col2.push(EQ("**h** = **g** ⊙ **h**_{img} + (1 − **g**) ⊙ **h**_{txt}", 2));
col2.push(P("A two-layer MLP maps **h** to the 128-dimensional embedding **f**, which is L2-normalised. Linear heads also give per-modality embeddings **z**_{img} and **z**_{txt} in the same space. When a modality is absent at query time, its feature is replaced by a learned null vector, so text-only and photo-only queries use the same model (Fig. 1)."));
col2.push(...FIG(path.join(__dirname, "architecture.png"), 3.45, 1, "Proposed model. Both encoders are trained from scratch; the gated fusion produces one product embedding used for all query types."));
col2.push(H2("D", "Training Objectives"));
col2.push(P("The main objective is the supervised contrastive loss [26] on the fused embedding, with the product type (and, at half weight, type + colour) as labels:"));
col2.push(EQ("ℒ_{sup} = SupCon(**f**, *y*_{type}) + 0.5·SupCon(**f**, *y*_{type+colour})", 3));
col2.push(P("A symmetric image–text contrastive loss (InfoNCE, as in [16]) aligns **z**_{img} and **z**_{txt} of the same product. To counter modality imbalance, **uni-modal supervision** applies the supervised contrastive loss to each modality separately, so that neither encoder can rely on the other [25]:"));
col2.push(EQ("ℒ = ℒ_{sup} + ℒ_{itc} + λ_{uni}[SupCon(**z**_{img}) + SupCon(**z**_{txt})]", 4));
col2.push(P(`with λ_{uni} = 0.5. We also evaluate two alternatives on the same model: OGM gradient modulation [24] adapted to contrastive training (the gradients of the dominant encoder are scaled by 1 − tanh(α(ρ − 1)), where ρ is the ratio of modality strengths; α = ${D.alpha.alpha} chosen on validation), and **noise-aware training**, in which 50% of training titles are damaged (30–100% of words removed, or replaced by another product's title) on the fusion input while labels stay correct, teaching the gate when not to trust text.`));

// IV. EXPERIMENTAL SETUP
col2.push(H1("IV", "Experimental Setup"));
col2.push(H2("A", "Dataset"));
col2.push(P(`We use the Fashion Product Images (Small) catalog [33]. Product types with fewer than 20 items are removed, leaving ${nProd.toLocaleString("en-US")} products in ${D.n_types} types. A stratified 70/15/15 split gives ${D.split.train.toLocaleString("en-US")} training, ${D.split.val.toLocaleString("en-US")} validation and ${D.split.test.toLocaleString("en-US")} test products; test products are never used for training, tuning or model selection. Photos are padded to a white square and resized to 64×64.`));
col2.push(H2("B", "Baselines"));
col2.push(P("All baselines share the same encoders, losses and training budget: an *image-only* CNN, a *text-only* Transformer, *early fusion* (concatenation + MLP), *late fusion* (equal-weight sum of image and text similarities) and *cross-attention fusion* (title tokens attend over the 8×8 image regions)."));
col2.push(H2("C", "Evaluation"));
col2.push(P("We report NDCG@10 [32] on the test products for four tasks: (i) *similar products*: each product queries the rest of the catalog and results of the same type are relevant; (ii) the same with type *and* colour; (iii) *text search* with shopper queries built from metadata (e.g. “navy blue shirts for men”), relevant = same colour, type and gender; and (iv) *photo search*, where the photo alone is the query. Results are the mean ± standard deviation over three seeds; differences are tested with a paired bootstrap over queries [35]."));
col2.push(H2("D", "Implementation"));
col2.push(P(`The models are implemented in PyTorch [34] and trained on one NVIDIA T4 GPU (Kaggle) for ${S.epochs} epochs with batch size 256, mixed precision, a one-epoch warm-up and a cosine learning-rate schedule; the epoch with the best validation score is kept. For each optimizer the learning rate is tuned on validation: SGD (Nesterov momentum 0.9) ${S.tuned_lr.SGD}, Adam ${S.tuned_lr.Adam}, AdamW ${S.tuned_lr.AdamW} (weight decay 0.05). ${S.best_optimizer}, selected on validation, is used for all method comparisons. One run takes about 6 minutes.`));

// V. RESULTS
col2.push(H1("V", "Results and Discussion"));
col2.push(H2("A", "Comparison of Fusion Methods"));
const order = ["image", "text", "early", "late", "xattn", "gated"];
const nm = { image: "Image-only CNN", text: "Text-only Transformer", early: "Early fusion", late: "Late fusion", xattn: "Cross-attention", gated: "Gated fusion" };
col2.push(...TCAP("II", "Test NDCG@10 of Fusion Methods (Mean ± Std, 3 Seeds)"));
col2.push(TABLE2(["Model", "Similar (type)", "Type + colour", "Text search", "Photo search"],
  [...order.map((m) => ({ cells: [nm[m], ...K.map((k) => ms(M[m], k))] })),
   { cells: ["Gated + uni-modal (ours)", ...K.map((k) => ms(G, k))], bold: true, shade: true }],
  [1200, 960, 960, 960, 960], [0], 13));
col2.push(P(`Table II shows that adding titles lifts similar-product accuracy from ${f3(M.image["i2i mean"])} (image only) to about ${f3(M.gated["i2i mean"])}. Plain gated fusion is significantly better than early and late fusion on similar products (${pstr(byKey(D.sig.filter(r => r.proposed === "gated" && r.vs === "late"), "metric", "i2i").p_value)} vs. late fusion) and than the image-only model on photo search. The text-only model is strong on type + colour (${f3(M.text["i2i_strict mean"])}) because titles literally contain type and colour words, and late fusion is best for text search (${f3(M.late["text2item mean"])}). With uni-modal supervision, the gated model becomes the best model for similar products (${f3(G["i2i mean"])}) and photo search (${f3(G["image2item mean"])}).`, { }));
col2.push(...FIG(path.join(ROOT, "results/scratch/method_comparison.png"), 3.45, 2, "Test NDCG@10 of the six fusion methods for the four tasks (mean ± std over three seeds)."));

col2.push(H2("B", "Optimizer Comparison"));
col2.push(...TCAP("III", "Optimizer Comparison on Gated Fusion (Test NDCG@10)"));
col2.push(TABLE2(["Optimizer", "LR", "Similar (type)", "Type + colour", "Text search", "Photo search"],
  ["SGD", "Adam", "AdamW"].map((o) => ({ cells: [o, String(O[o].lr), ...K.map((k) => ms(O[o], k))], shade: o === S.best_optimizer })),
  [760, 460, 950, 950, 950, 970], [0], 14));
col2.push(P(`All three optimizers reach nearly the same accuracy (within about 0.01, Table III), and their training curves converge similarly (Fig. 3). SGD is best on type + colour and photo search, AdamW on text search. Adam and AdamW behave almost identically, because decoupled weight decay has little effect over 30 short epochs. Training the model at all matters far more than the choice of optimizer.`));
col2.push(...FIG(path.join(ROOT, "results/scratch/optimizer_curves.png"), 3.45, 3, "Training loss and validation NDCG@10 of the gated model for SGD, Adam and AdamW."));

col2.push(H2("C", "Ablation Study"));
const ablOrder = [["full", "Gated fusion (full)"], ["no_supcon", "− supervised contrastive"], ["no_itc", "− image–text contrastive"], ["no_consistency", "− consistency loss"], ["no_modality_dropout", "− modality dropout"]];
col2.push(...TCAP("IV", "Ablation of the Gated Model (Test NDCG@10)"));
col2.push(TABLE2(["Variant", "Similar (type)", "Type + colour", "Text search", "Photo search"],
  ablOrder.map(([k, n]) => ({ cells: [n, ...K.map((m) => f3(A[k][`${m} mean`]))] })),
  [1400, 910, 910, 910, 910], [0], 13));
col2.push(P(`Removing the supervised contrastive loss causes the largest drop (similar products ${f3(A.full["i2i mean"])} → ${f3(A.no_supcon["i2i mean"])}). Removing the image–text contrastive loss mainly hurts text search (${f3(A.full["text2item mean"])} → ${f3(A.no_itc["text2item mean"])}), confirming that aligning the two modalities is what enables cross-modal queries. Two components we initially proposed, modality dropout and a missing-modality consistency loss, did not help on clean data and are reported here for completeness.`));

col2.push(H2("D", "Countering Modality Imbalance"));
const bn = { base: "Gated fusion (original)", ogm: `+ OGM (α = ${D.alpha.alpha})`, uni: "+ uni-modal supervision", noise: "+ noise-aware training", noise_uni: "+ noise-aware + uni-modal" };
col2.push(...TCAP("V", "Modality-Balancing Variants of the Gated Model"));
col2.push(TABLE2(["Variant", "Similar (type)", "Type + colour", "Text search", "Photo search"],
  ["base", "ogm", "uni", "noise", "noise_uni"].map((v) => ({
    cells: [bn[v], ...K.map((k) => { const s = v === "base" ? null : psig(v, k); const mark = s && s.p_value < 0.05 ? (s.diff > 0 ? " ▲" : " ▼") : ""; return ms(B[v], k) + mark; })],
    bold: v === "uni", shade: v === "uni" })),
  [1200, 960, 960, 960, 960], [0], 13));
col2.push(P(`▲/▼ in Table V mark significant improvements/drops versus the original gated model (paired bootstrap, p < 0.05). Uni-modal supervision improves **all four tasks significantly** (${K.every((k) => psig("uni", k).p_value < 0.001) ? "p < 0.001 for each" : K.map((k) => pstr(psig("uni", k).p_value)).join(", ")}): similar products ${f3(B.base["i2i mean"])} → ${f3(G["i2i mean"])}, type + colour ${f3(B.base["i2i_strict mean"])} → ${f3(G["i2i_strict mean"])}, text search ${f3(B.base["text2item mean"])} → ${f3(G["text2item mean"])} and photo search ${f3(B.base["image2item mean"])} → ${f3(G["image2item mean"])}. Giving each encoder its own learning signal prevents the title branch from doing all the work. OGM gives only small gains, and noise-aware training lowers clean accuracy, as expected for a model trained on deliberately damaged titles.`));

col2.push(H2("E", "Robustness to Missing and Wrong Titles"));
const conds = [["clean", "Clean"], ["drop 30%", "30% words removed"], ["drop 60%", "60% removed"], ["no title", "No title"], ["wrong 20%", "20% wrong"], ["wrong 50%", "50% wrong"]];
const rm = [["image", "Image-only"], ["text", "Text-only"], ["late", "Late fusion"], ["gated", "Gated"], ["gated+uni", "Gated + uni (ours)"], ["gated+noise+uni", "+ noise-aware (ours)"]];
col2.push(...TCAP("VI", "Similar-Product NDCG@10 under Damaged Titles"));
col2.push(TABLE2(["Model", ...conds.map((c) => c[1])],
  rm.map(([m, n]) => ({ cells: [n, ...conds.map(([c]) => f3(rob(m, c)))], bold: m === "gated+noise+uni", shade: m.startsWith("gated+") })),
  [1360, 610, 620, 620, 610, 610, 610], [0], 13));
col2.push(P(`To test catalogs with poor titles, we damage the titles of the test products (photos untouched) and re-evaluate; each condition is averaged over three random draws (Table VI, Fig. 4). Without titles, the text-only model collapses to ${f3(rob("text", "no title"))}, whereas the fused models keep ${f3(rob("gated", "no title"))}–${f3(rob("gated+noise+uni", "no title"))} because the photo carries the recommendation. Plain fusion, however, over-trusts damaged text: at 60% of words removed the gated model falls to ${f3(rob("gated", "drop 60%"))}, below the image-only model (${f3(rob("image", "drop 60%"))}). Noise-aware training removes this weakness: it is the most robust model in every damaged condition (${f3(rob("gated+noise+uni", "drop 60%"))} at 60% removed, ${f3(rob("gated+noise+uni", "wrong 50%"))} with 50% wrong titles) and never falls below image-only, at the cost of lower clean accuracy (${f3(rob("gated+noise+uni", "clean"))}). A deployment can therefore choose the clean-accuracy model or the robust model according to catalog quality.`));
col2.push(...FIG(path.join(ROOT, "results/scratch/robustness.png"), 3.45, 4, "Similar-product NDCG@10 when title words are removed (left) or titles are wrong (right)."));

col2.push(H2("F", "Demonstration System and Product Generation"));
col2.push(P(`The final model is deployed in a Streamlit web application that answers similar-product, text, photo and photo + text queries on the ${nTest.toLocaleString("en-US")} unseen products in real time on a laptop CPU, since the model has only 3.8 M parameters. As a complementary experiment, a conditional DCGAN [30] trained from scratch with DiffAugment [31] generates new 64×64 product images per category; after fixing an initial mode collapse, its Fréchet distance in CLIP feature space improved from ${D.gan["FD-CLIP untrained generator (baseline)"].toFixed(2)} (untrained generator) to ${D.gan["FD-CLIP (generated vs real)"].toFixed(2)}, and ${Math.round(100 * D.gan["class accuracy (generated)"])}% of generated images are recognised as the requested category (chance ${Math.round(100 * D.gan["chance accuracy"])}%). Image quality remains limited by the small CPU training budget.`));

// VI. LIMITATIONS AND CONCLUSION
col2.push(H1("VI", "Limitations"));
col2.push(P("Relevance is derived from product metadata rather than real user interactions, and product titles in this catalog contain the type and colour words, which favours text. Images are small (64×64) to keep training cheap. The noise rate for noise-aware training (50%) was not tuned; a lower rate may retain more clean accuracy. Pretrained models such as CLIP [16] are expected to be more accurate but were deliberately excluded to study fusion under from-scratch training."));
col2.push(H1("VII", "Conclusion and Future Work"));
col2.push(P(`We presented a multimodal product recommender whose image and text encoders are trained entirely from scratch and fused by a gated unit with uni-modal supervision. On ${nTest.toLocaleString("en-US")} unseen products it reaches NDCG@10 of ${f3(G["i2i mean"])} for similar products and ${f3(G["image2item mean"])} for photo search, significantly improving over plain gated fusion on all four tasks. A controlled comparison showed that fusion design matters more than the optimizer, and a new noisy-title benchmark showed that fusion keeps recommendations working when titles are missing and that noise-aware training makes it robust to damaged titles. Future work includes tuning the noise rate, adaptive trust in the title at inference time, larger images, and evaluation with real user interaction data.`));

// References
col2.push(new Paragraph({ children: [new TextRun({ text: "References", font: FONT, size: SZ, smallCaps: true })],
                         alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 } }));
REFS.forEach((r, i) => col2.push(new Paragraph({
  children: [new TextRun({ text: `[${i + 1}]`, font: FONT, size: 16 }), new TextRun({ text: "\t", font: FONT, size: 16 }), ...runs(r, { size: 16 })],
  tabStops: [{ type: TabStopType.LEFT, position: 400 }], indent: { left: 400, hanging: 400 },
  alignment: AlignmentType.LEFT, spacing: { after: 30 },
})));

// ---------------------------------------------------------------- document
const page = { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1440, left: 900, right: 900 } };
const doc = new Document({
  creator: "[Student Name]",
  title: "Gated Multimodal Fusion with Uni-Modal Supervision for Product Recommendation",
  styles: { default: { document: { run: { font: FONT, size: SZ } } } },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
                                                             style: { paragraph: { indent: { left: 280, hanging: 200 } } } }] }] },
  sections: [
    { properties: { page }, children: [...title, authors, new Paragraph({ children: [], spacing: { after: 200 } })] },
    { properties: { page, type: SectionType.CONTINUOUS, column: { count: 2, space: 360, equalWidth: true } }, children: col1 },
    { properties: { page, type: SectionType.CONTINUOUS }, children: litSection },
    { properties: { page, type: SectionType.CONTINUOUS, column: { count: 2, space: 360, equalWidth: true } }, children: col2.filter(Boolean) },
  ],
});
const out = path.join(ROOT, "Multimodal_Product_Recommendation_IEEE_Report.docx");
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(out, b); console.log("written", out); });
