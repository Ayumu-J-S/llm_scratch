# 改善チケットのレビュー・計測チェック

この文書は、`ROADMAP.md` のチケットを実装したあとに「動いた」だけで完了扱いせず、
実際の学習系として健全かをレビューするための忘れ防止カタログである。

これは全項目を毎回実行するチェックシートではない。また、新しい単体テストや形式的な
ソフトウェアテストを一律に要求する文書でもない。変更内容に応じて必要な観点だけを選び、
実データ、実学習経路、DGX Spark 上の短い計測、ログやプロファイルから判断する。
`ROADMAP.md` に個別の acceptance criteria や validation がある場合、それらは別途満たす。

## 1. 使い方

レビューは次の順番で行う。

1. チケットの目的、変更前の問題、意図した挙動を一文ずつ確認する。
2. 「変更別ルーター」から必要なチェック領域だけを選ぶ。
3. 性能や資源に影響し得る変更だけ、比較可能な短い計測を行う。
4. 結果を `PASS`、`PASS WITH NOTE`、`FAIL` のいずれかで残す。
5. `N/A` にした重要領域には、不要な理由を一言残す。

判定の意味は次のとおり。

- `PASS`: 意図した改善が確認でき、正しさ、学習健全性、性能、運用性に未説明の悪化がない。
- `PASS WITH NOTE`: 意図した改善は確認できる。既知のトレードオフや後続課題があり、数値と理由が記録されている。
- `FAIL`: 学習目的が変わる、データが詰まる、GPU経路が使われない、資源枯渇が見込まれる、再現不能、または変更が次の実験を不必要に難しくする。

### 常に見る最小セット

どのチケットでも、少なくとも次の5点だけは確認する。

- [ ] チケットの目的に直接対応する証拠がある。変更量やコードの見た目だけを証拠にしていない。
- [ ] 学習目的、対象トークン、データ境界、評価境界のどれかが意図せず変わっていない。
- [ ] 実際に使うコマンドと解決済みHydra設定で確認している。別のデバッグ経路だけで確認していない。
- [ ] 失敗を成功に見せる暗黙動作がない。特にCUDAからCPU、real profileからfixture、onlineから未記録runへの暗黙fallbackがない。
- [ ] チケット外の仕組み、設定分岐、互換レイヤー、実行経路を増やしていない。増やした場合は必要性が説明されている。

## 2. 変更別ルーター

該当する行だけを選ぶ。複数領域にまたがるチケットでは行を組み合わせる。

| 変更したもの | 主に見る領域 | 通常の確認規模 |
| --- | --- | --- |
| 文書、PRテンプレート、記録形式だけ | 実験・引き継ぎ、変更容易性 | 静的レビューとdry-run |
| Hydra設定、CLI、entrypoint | 実行経路、再現性、暗黙fallback、変更容易性 | config composeと短いsmoke |
| 依存関係、コンテナ、PyTorch/CUDA | DGX環境、実GPU、precision、再現可能な導入 | CUDA 10〜100 step |
| tokenizer | データ供給、圧縮率、語彙由来のモデルサイズ、offline再現性 | frozen corpus計測と実batch |
| dataset、manifest、split、filter | データ同一性、供給速度、長文tail、cache、split漏洩 | loader単体と短いend-to-end |
| packing、collate、shuffle、cursor | 対象transition、token accounting、順序、resume、供給速度 | fixture確認と短いend-to-end |
| model、attention、embedding、loss | 数学的挙動、学習健全性、step time、GPU kernel、memory | CPU smokeとDGX 50〜200 step |
| optimizer、scheduler、AMP、accumulation | update意味、数値安定性、effective tokens、GPU効率 | DGX 100 step前後 |
| training loop、logging | 同期点、step揺れ、停止条件、local evidence、長期安定性 | profilerを含むDGX短時間run |
| checkpoint、resume | pause時間、書込量、atomicity、軌跡、disk予測 | 中断・再開fixtureと実サイズ推計 |
| validation、benchmark、generation | 学習との分離、checkpoint同一性、評価時間、再現性 | 固定checkpointで比較 |
| W&B、artifact | hot pathのoverhead、障害分離、容量、offline動作 | disabled/offline/online比較 |
| performance optimization | objective同一性、品質許容差、再現可能な速度差、rollback | 基準実装とのA/Bを3回以上 |
| 本番baseline run | この文書の全領域から該当項目を選択 | preflight、短いpilot、計画run |

### 確認規模の目安

- `R0 — review`: 文書、設定、依存方向、差分だけを見る。性能主張はしない。
- `R1 — smoke`: CPUまたは小fixtureで1〜10 step。配線と明白な失敗だけを見る。
- `R2 — target smoke`: DGX Spark、実precision、実データ経路で50〜200 optimizer step。定常性能と短期安定性を見る。
- `R3 — pilot`: 15〜60分程度の代表run。温度、clock、network tail、memory creep、checkpoint/eval込みの挙動を見る。
- `R4 — consequential run`: 事前宣言した時間・token予算のrun。モデル品質や研究上の結論に使う。

小さな設定・文書チケットに `R2` 以降を強制しない。一方、データ、モデル、training loop、
precisionを変えたのに `R1` だけで「速い」「GPUを使えている」と結論しない。

## 3. 比較計測の共通ルール

### 比較条件

- [ ] 比較対象のbaseline commitまたはrun IDを記録する。
- [ ] hardware、OS、driver、CUDA、PyTorch、precisionを揃える。
- [ ] model config、tokenizer、sequence length、micro batch、gradient accumulation、対象token数を揃える。
- [ ] data manifest、source比率、cache状態、network条件を揃える。
- [ ] W&B、profiler、model watch、checkpoint、validationの有効・無効を揃える。
- [ ] compileやkernel autotuneのwarmupを測定区間から外す。
- [ ] CUDA時間を個別に測る場合はeventまたは測定境界のsynchronizeを使う。通常の学習hot pathへ毎step同期を追加しない。
- [ ] 短いrunは原則3回以上行い、単一の最良値ではなく中央値とばらつきを残す。
- [ ] loaderは必要に応じてcold cacheとwarm cacheを分ける。両者を混ぜて平均しない。

### 最低限残す数値

性能に影響するチケットでは、該当するものを残す。

- optimizer step timeのmedian、p95、最大値
- target tokens/s。入力tokenではなくlossを持つ非paddingの次token target数を基準にする
- data wait timeまたは`next(loader)`でblockした時間と、step全体に対する割合
- forward、backward、optimizer、logging、checkpoint、validationの時間内訳
- GPU active/utilizationの時系列、clock、power、temperature
- CPU使用率、load、memory available、swap in/out、disk read/write、network受信
- PyTorch peak allocated/reserved memoryと、DGX Spark全体のavailable memory
- loss、token-weighted NLL、learning rate、gradient norm、非finite数
- model parameter数、checkpointサイズ、run全体のdisk予測

### 既定の調査トリガー

以下は普遍的な合格値ではなく、数値を見落とさないための既定トリガーである。チケットで別の
予算を事前宣言していればそちらを使う。

- baseline比でmedian tokens/sが5%以上低下: 原因を調べて記録する。
- baseline比でmedian tokens/sが10%以上低下: 意図した品質・安全性との交換でなければ原則`FAIL`。
- p95 step timeがmedianの1.5倍を継続的に超える: data、logging、checkpoint、thermal、network tailを分解する。
- 定常区間のdata waitがstep時間の5%以上: loader headroomを確認する。
- 定常区間のdata waitが10%以上、またはGPU timelineに反復する空白: 原因が分かるまで長時間runを開始しない。
- loader-onlyのtarget token供給がend-to-end消費の1.2倍未満: データ経路に余裕なしとみなす。
- warmup後にmemory使用がstepとともに単調増加する、またはswap in/outが続く: 長時間runを開始しない。
- NaN/Inf、token数不一致、split重複、checksum不一致、resume後の意図しない順序差: 許容差なしで`FAIL`。

GPU utilizationの単一平均値だけを合否にしない。短いkernelやmemcpyが走っているだけでも
「利用中」に見え、逆に小モデルはGPUを飽和できないことがある。tokens/s、GPU timeline、
kernel、data wait、clockと併せてボトルネックを判断する。

## 4. データ供給とtokenizer

### 4.1 データがGPUを待たせていないか

- [ ] `model-only`、`loader-only`、`end-to-end` の3つを分けて測る。
- [ ] `model-only` は同じshapeのdevice上synthetic batchを再利用し、モデル側の上限を見る。
- [ ] `loader-only` は実source、実tokenizer、実packingで一定のtarget tokensを消費し、供給上限を見る。
- [ ] `end-to-end` は実学習経路でdata waitとGPU idle gapを見る。
- [ ] loader-onlyが遅い場合、source read、network、JSON decode、tokenize、pack、IPC、collate、host-to-deviceのどこかを分ける。
- [ ] 平均だけでなく文書長p50/p95/p99付近を通し、長文一件でproducerが止まらないかを見る。
- [ ] prefetch queueが空の時間と満杯の時間を観測する。常に空ならproducer不足、常に満杯ならbuffer増加は効かない。
- [ ] prefetch on/offを同じ順序・同じtoken数で比べ、速度差とCPU/memory増加を確認する。
- [ ] startup、最初のbatch、定常batch、source切替、再接続の時間を分ける。
- [ ] validation、resume直後、cache missでも許容できる停止時間かを見る。

### 4.2 tokenization

- [ ] 日本語、英語、混在文、ASCII記号、絵文字、壊れたUnicode候補でfallbackや例外の件数を出す。
- [ ] tokens/character、tokens/UTF-8 byte、sequence length p50/p95/p99を言語別に比較する。
- [ ] docs/s、characters/s、tokens/s、1文書のp95/p99 latency、peak RSSを測る。
- [ ] tokenizer throughputが学習の必要target tokens/sを十分上回る。
- [ ] 長文を丸ごとtokenizeする設計なら、最大想定文書でlatencyと一時memoryを確認する。
- [ ] tokenizer artifact、revision、special token IDs、vocab sizeがofflineで同一になる。
- [ ] encode/decodeのround-trip可否と、loss計算に使うPAD/EOS/BOSの意味が記録されている。
- [ ] 外部tokenizerと一緒にpretrained model weightsやchat templateを誤って導入していない。
- [ ] 語彙変更によるembeddingとLM headのparameter数、optimizer state、checkpoint、step time増加を計算する。
- [ ] 圧縮率の改善が語彙由来の出力層コスト増に見合うか、tokens/sだけでなく文字またはbyte/sでも比較する。

### 4.3 packing、境界、token accounting

- [ ] 連続streamで学習対象にするnext-token transitionが欠落・重複していない。
- [ ] 文書境界、source境界、quotaで切れた断片に、意図したEOSまたは明示的な境界規則がある。
- [ ] emitted token、input token、target token、dropped remainder、paddingの各数が区別される。
- [ ] `max_tokens` がtokenizer出力、target tokens、optimizer予算のどれを指すか一意である。
- [ ] source比率は文書数ではなく、宣言した基準の実現値で確認する。
- [ ] partial batchや短い最終windowがmetricを歪めない。
- [ ] buffer操作が長文サイズに対して二次的に遅くならない。
- [ ] metadata/source spanを有効にしたときも境界が正しく、overheadが許容範囲である。

### 4.4 source、cache、network

- [ ] train/validationのdocument IDとnormalized content hashに重複がない。
- [ ] source、revision、config、split、license/terms、checksumがrun前に確定する。
- [ ] source順序変更やprefetch変更でsplit membershipが変わらない。
- [ ] remote sourceのtimeout、retry、backoffが無限停止やsilent skipを起こさない。
- [ ] source別のread throughput、retry数、rejection数、欠損率を記録する。
- [ ] cache hit/miss、eviction、download中断、破損、満杯時の挙動が観測できる。
- [ ] cache上限、checkpoint予測、run logs、一時file、OS用headroomの合計がdisk空き容量内に収まる。
- [ ] cache keyがURLだけなら、同じURLで内容が変化しない保証またはcontent checksumがある。
- [ ] remote sourceが落ちても、既存checkpointやmanifestを壊さず失敗理由を残せる。

## 5. GPU、DGX Spark、システム資源

### 5.1 実GPU経路

- [ ] Hydraで要求deviceが明示され、`cuda`要求時に利用不能ならdata load前に失敗する。
- [ ] run logにGPU名、compute capability、driver、CUDA runtime、PyTorch build、BF16 capabilityが残る。
- [ ] 実行processが`nvidia-smi`または対応monitorにcompute processとして現れる。
- [ ] model parameter、input、label、lossが期待deviceにあり、CPU上の主要演算へ落ちていない。
- [ ] 代表stepのforward、backward、optimizerにCUDA kernelが存在する。
- [ ] CPU smokeは明示的なprofileだけで許可し、real profileはsilent CPU fallbackしない。
- [ ] aarch64向けpackage/containerであり、x86_64限定wheelやemulationに依存していない。

### 5.2 GPUが有効に使われているか

- [ ] warmup後の連続区間を測り、初期化時間を定常性能に混ぜない。
- [ ] step間のGPU idle gapと、その時のCPU thread状態、data wait、loggingを対応させる。
- [ ] low utilizationが小batch、短sequence、CPU同期、data starvation、kernel launch過多、memory boundのどれかを特定する。
- [ ] micro batchを増やしたときtokens/sが伸びるか、memory ceilingまで数点測る。
- [ ] sequence length変更でattentionコストとmemoryが想定どおり変わるかを見る。
- [ ] BF16/FP32を同じwork量で比較し、速度、memory、loss、non-finite、実kernel dtypeを確認する。
- [ ] Tensor Coreを使う想定なら、shape、dtype、実kernelがその想定と一致する。
- [ ] `.item()`、頻繁なsynchronize、blocking copy、per-step artifact/loggingがGPU queueを止めていない。
- [ ] profiler自体のoverheadを通常runの数字に混ぜない。

### 5.3 DGX Spark固有の確認

DGX SparkはGrace Blackwell、ARM64、128 GBのCPU/GPU共有memoryを持つUMA機である。
dedicated VRAM機と同じ見方をしない。

- [ ] GPU memoryだけでなく、system available memory、process RSS、PyTorch allocator、page fault、swapを一緒に見る。
- [ ] `nvidia-smi`のMemory-Usageが未対応または小さく見えても「余裕あり」と判断しない。
- [ ] warmup後にswapが増え続けない。swap発生時はstep time tailとUMA page migrationを確認する。
- [ ] CPU tokenization、file cache、GPU tensor、optimizer stateが同じ128 GBを競合する前提でheadroomを決める。
- [ ] peakだけでなく、checkpoint保存、validation、prefetch bufferが重なる最悪時memoryを測る。
- [ ] supplied 240W power supplyを使い、power/thermal cap、clock低下、temperature上昇がないかpilotで見る。
- [ ] 15〜60分pilotで、最初の100 stepだけでは見えないthermal throttlingとclock変動を確認する。
- [ ] desktop、ブラウザ、Jupyter、別のCUDA processなど、比較run間のbackground負荷を揃える。
- [ ] root filesystem上のdataset/cache/checkpoint競合とNVMe I/O待ちを確認する。
- [ ] OS/driver/firmware更新前後の性能を同一baselineで再計測し、環境変更をrun identityに含める。

### 5.4 memoryとstorageの判定

- [ ] model parameter、gradient、optimizer state、activation、batch、prefetch、tokenizer、file cacheを別々に概算する。
- [ ] `torch.cuda.max_memory_allocated()`だけを全体memoryとして扱わない。
- [ ] warmup後のpeakが安定し、step数に比例して増えない。
- [ ] OOM時にbatchを暗黙縮小して続行しない。失敗したconfigを保存する。
- [ ] checkpoint一個の実サイズ、回転個数、一時atomic-write分、best/final/milestone分を合計する。
- [ ] full run終了時のcache、checkpoints、logs、profilesの予測値と安全headroomをrun前に出す。
- [ ] OSやcheckpoint領域を侵食するcache設定を拒否する。

## 6. 学習loopと数値健全性

### 6.1 学習しているものが同じか

- [ ] batchの`inputs`、`labels`、mask、PAD無視、EOSの意味が変更前後で意図どおりである。
- [ ] loss reductionとtoken weightingが明示され、partial batchで意味が変わらない。
- [ ] gradient accumulation中のloss scaleとoptimizer stepの意味が正しい。
- [ ] effective target tokens/updateが設定と実測で一致する。
- [ ] schedulerがmicro stepではなく意図したoptimizer step/tokenで進む。
- [ ] skipped update、overflow、clipping後もcounterとschedulerがずれない。
- [ ] dropout、train/eval mode、seedの切替がvalidation後やresume後に壊れない。

### 6.2 数値安定性

- [ ] loss、perplexity、gradient norm、parameter/update norm、learning rateが定期的に見える。
- [ ] NaN/Infを発生したstep、batch identity、直前checkpointを残して停止できる。
- [ ] 初期lossがvocab sizeやdata分布から見て不自然でない。
- [ ] lossが下がるだけでなく、gradientがゼロ張り付き、爆発、特定layerだけ異常になっていない。
- [ ] clipping率が高すぎて恒常的にupdateを変えていない。
- [ ] BF16でFP32 baselineと短期loss trajectoryが許容範囲にあり、必要なreduction/stateはFP32である。
- [ ] perplexityの`exp(loss)`がoverflowして監視自体を壊さない。
- [ ] tokenizer、data、modelを変えたrun同士のraw lossを無条件に同じ尺度として扱わない。

### 6.3 stepの内訳と同期点

- [ ] `next(loader)`、host/device準備、forward、loss、backward、clip、optimizer、scheduler、metricsの時間を分ける。
- [ ] `loss.item()`やgradient norm取得のためのdevice同期頻度を把握する。
- [ ] W&Bやconsole logを毎stepから間引いたA/Bで、hot path overheadを測る。
- [ ] tqdm、JSON書込、manifest hash、checkpoint verificationが定常stepに入り込んでいない。
- [ ] validationとcheckpoint cadenceが独立し、それぞれの停止時間を予算化できる。
- [ ] event処理のoff-by-oneで余分なvalidation/saveやtoken超過が起きない。

### 6.4 15〜60分pilotで見ること

- [ ] tokens/s、step time、GPU clock、temperature、memoryが定常化する。
- [ ] lossとgradient normが有限で、突然のspikeが説明できる。
- [ ] source比率とrejection率が時間とともに大きく漂わない。
- [ ] network retry、queue starvation、disk I/O waitが周期的に発生しない。
- [ ] checkpointとvalidation後に元のthroughputへ戻る。
- [ ] ETAが実測tokens/s、validation、checkpoint overheadを含めた値と整合する。
- [ ] stop条件で安全に止まり、次に再開または診断できる証拠が残る。

## 7. モデル改良のしやすさとソフトウェアの健全性

ここでは抽象化の多さを評価しない。次の実験を小さく安全に行えるかを評価する。

### 7.1 変更面積

- [ ] model変更がdata loader、checkpoint policy、W&B実装の書き換えを要求しない。
- [ ] tokenizer変更が複数の独立したtokenizer選択箇所を同期修正させない。
- [ ] data source変更がtrainerやmodelの分岐追加を要求しない。
- [ ] optimizer/scheduler/precisionの科学的な選択はHydraにあり、source定数になっていない。
- [ ] 実装詳細まで全部config化せず、実験上意味のある選択だけを公開している。
- [ ] 既存のlocal/streaming、train/eval/generate経路で同じ責務の実装を重複させていない。
- [ ] entrypointは組立てに集中し、data、model、training、evaluation、checkpointの責務が混ざっていない。

### 7.2 依存方向と置換可能性

- [ ] trainerは特定model classの内部構造ではなく、明示されたforward/loss contractに依存する。
- [ ] modelはW&B、Hydra、dataset、filesystemをimportしない。
- [ ] data層はmodelやoptimizerをimportしない。
- [ ] checkpointは再構築に必要なidentityを持つが、不要なruntime objectをpickleしない。
- [ ] optional integrationのimport失敗が、その機能を使わないoffline/CPU経路を壊さない。
- [ ] performance専用コードは小さい境界の後ろにあり、reference挙動と比較できる。

### 7.3 このリポジトリの方針

- [ ] runtime/training設定はHydraに置き、別の`config.py`を追加していない。
- [ ] direct importを使い、不要な再exportやservice locatorを増やしていない。
- [ ] 現在必要のないcompatibility alias、deprecated path、shimを残していない。
- [ ] 一つのfocused ticketに関係しないarchitecture変更を混ぜていない。
- [ ] notebookだけにcanonicalな処理を置いていない。
- [ ] 実際に測定したbottleneckなしでcustom kernel、compile、分散化を追加していない。

### 7.4 レビュー時の「次の変更」思考実験

コードを追加せず、次の質問に答えて変更面積を確認する。

- normalizationをLayerNormから別実装へ変えるなら、どこを変更するか。
- positional representationだけを変えるなら、trainer/dataに触れずに済むか。
- tokenizerを差し替えるなら、train、stream、generation、model vocabが一箇所のidentityから揃うか。
- dataset混合比を変えるなら、Hydraとmanifestだけで再現可能か。
- 1時間runを24時間runへ伸ばすなら、source編集なしでtoken/save/eval予算を変更できるか。
- 中断runを別configで誤resumeしたとき、開始前に拒否できるか。

答えが多数の無関係なfileや手作業に広がるなら、チケットが新しい結合を持ち込んでいないかを確認する。
ただし、将来の可能性だけを理由にframeworkやplugin systemを先回りして作らない。

## 8. 再現性、研究integrity、評価

### 8.1 run identity

- [ ] Git SHA、dirty state、resolved Hydra config、lock/container identityがある。
- [ ] hardware、OS、driver、CUDA、PyTorch、precision、seedがある。
- [ ] model、tokenizer、train/validation manifestのfingerprintがある。
- [ ] target tokens、optimizer steps、elapsed timeの開始・終了counterがある。
- [ ] W&B無効時にもrun directoryだけで同じ入力を特定できる。
- [ ] PyTorch/platformをまたぐbitwise同一性を無理に主張せず、必要な再現レベルを記録する。
- [ ] deterministic設定による性能低下がある場合、その差を別測定で示す。

### 8.2 学習データと評価データ

- [ ] trainとvalidationのidentity/content重複がない。
- [ ] benchmark devとreserved testのaccess pathが分離される。
- [ ] benchmark contaminationのexact/normalized照合結果がある。
- [ ] external modelのweights、logits、生成出力がtraining data/targetへ入っていない。
- [ ] tokenizer以外のpretrained capabilityをrandom-initialization modelへ持ち込んでいない。
- [ ] same-corpus memorizationをheld-out validationと呼んでいない。
- [ ] checkpoint、prompt、few-shot、decoding、scorer revisionが評価結果と結び付く。

### 8.3 結論の健全性

- [ ] hypothesisとsuccess/failure条件が結果を見る前に書かれている。
- [ ] baselineとの違いが一つの解釈可能な変更に絞られている。
- [ ] 速度改善時に、tokenizerやtarget数の変更で見かけ上速くなっていない。
- [ ] loss改善時に、data leakage、mask、reduction、vocab変更で尺度が変わっていない。
- [ ] 最良seedや最良checkpointだけを選んで一般化していない。
- [ ] negative resultにもconfig、run、失敗原因、次に否定できたことが残る。
- [ ] 「速くなった」「良くなった」ではなく、条件、差、ばらつき、trade-offを書く。

## 9. checkpoint、W&B、長時間運用

### 9.1 checkpointとresume

- [ ] model、optimizer、scheduler、precision、counter、RNG、stream cursorを保存する。
- [ ] resolved config、data/tokenizer/run identityを照合してからresumeする。
- [ ] 書込はtemporary path、read-back、atomic renameの順である。
- [ ] save pause、file size、write throughput、verification時間を実サイズで測る。
- [ ] save中の追加memory peakと、UMA/system memoryへの影響を見る。
- [ ] rotationは新checkpointの検証後だけ古いrecovery checkpointを消す。
- [ ] corrupt newestからprevious verifiedへ戻る挙動が明示される。
- [ ] resume直後のbatch/counter/LRがuninterrupted runの期待suffixと一致する。
- [ ] signal、例外、disk full、network断で最後のverified checkpointを壊さない。

### 9.2 W&Bとlogging

- [ ] W&B disabled/offlineでも学習、local metrics、checkpointが成立する。
- [ ] missing login、quota不明、network断がtraining hot pathを止めない方針がある。
- [ ] scalar cadenceを間引いたA/Bでoverheadを確認する。
- [ ] model watchは必要なrunだけで有効化し、通常はoffである。
- [ ] raw corpusや全checkpointをartifactとして上げない。
- [ ] artifact policyが`none|best|final|milestone`など明示的である。
- [ ] upload前にprojected size、現在のplan/usage/retentionを確認する。
- [ ] failed uploadをrun失敗と混同せず、local evidenceとretry情報を残す。
- [ ] metric key、step axis、token axisがrun間で比較可能である。

### 9.3 長時間runのpreflight

- [ ] 短い実data pilotが完了している。
- [ ] success、stop、failure条件がある。
- [ ] tokens/sからrun時間、validation、checkpoint、benchmark時間を予測している。
- [ ] memory、disk、cache、W&B artifactの最悪量を予測している。
- [ ] GPU、temperature、clock、data wait、loss、gradient、diskを監視できる。
- [ ] PID、run directory、command、start time、expected end timeが分かる。
- [ ] 異常runを止めてもconfig、logs、last verified checkpointを保存できる。
- [ ] retryは元runと原因にlinkし、失敗したrunを消して履歴を作り直さない。

## 10. 現在の実装で特に見落としやすい点

この表は現状コードから見える観測ポイントであり、すべてを今すぐ直す一覧ではない。
関連するチケットをレビューするときに確認する。

| 現在の箇所 | 見落としやすいリスク | 関連変更で見ること |
| --- | --- | --- |
| `src/train.py` のmodule-level `DEVICE` | CUDA不可でもCPUへsilent fallbackし、import時にdeviceが固定される | 明示Hydra device、data load前fail、実CUDA process |
| `src/train.py` のbatch `.to(device)` | pinned memoryを使ってもcopyがblockingで、computeと重ならない可能性 | H2D時間、`non_blocking` A/B、UMAでの実測 |
| `src/train.py` のpreview batch | real streamを一度起動して閉じ、trainingで先頭から再起動する | startup重複、worker終了、先頭batch再読込時間 |
| `StreamingTokenDataset` | `num_workers=0`で、独自prefetch processだけがproducer | loader headroom、CPU一core詰まり、順序を守るworker設計 |
| `StreamLoader._sample_iter` | 文書全体を同期tokenizeし、長文でproducer latency/memoryが跳ねる | 文書長p99、tokenize p99、peak RSS |
| `StreamLoader._packed_iter` | list先頭の反復削除が大きいbufferで高コストになり得る | 最大文書でCPU profile、buffer長とtokens/s |
| process prefetch | windowごとにNumPy配列をprocess queueでserializeする | IPC比率、buffer_size A/B、queue empty率 |
| streamのepoch再iteration | seedとsource iteratorが毎epoch先頭へ戻り、同じprefixを反復する | horizon、repeat policy、cursor、resume suffix |
| local text dataset | 全token tensorと重なる全windowをmap-styleでshuffleする | smoke専用か、real dataへ拡張していないか |
| `Trainer._train_epoch` | 毎stepの`loss.item()`とW&B logがCUDA同期・I/Oを作り得る | logging cadence A/B、timelineのstep間gap |
| `run.watch(self.model)` | gradient/parameter監視の量と頻度がhot pathを重くし得る | watch off/on差、W&B無効時のbaseline |
| epoch validation | streamingでは大きなvalidationをepochごとに全走査し得る | validation予算、token cadence、停止時間 |
| batch平均loss | 最終partial batchを同じ重みで平均し、token-weighted NLLとずれる | evaluated target token数と手計算NLL |
| raw model-only checkpoint | optimizer/counter/cursorなし、毎epoch同じfileへ同期保存 | resume可能性、pause、atomicity、回転、disk |
| external tokenizer導入 | vocab増加がembedding/LM head/optimizer/checkpointを大幅に増やし得る | parameter差、memory、step time、bytes/s |
| cache上限750 GB | 現在のroot空き約525 GBより大きく、OS/checkpoint headroomがない | dynamic preflight、安全上限、full-run予測 |
| DGX Spark UMA | `nvidia-smi`のdedicated VRAM表示だけでは全体圧迫を捉えられない | `free`/RSS/swap/PyTorch peakを併記 |

## 11. ROADMAPチケット別の選択ガイド

以下は各チケットで優先して選ぶ領域である。記載のない全項目を実行する必要はない。

| Ticket | 優先チェック | そのチケット固有で忘れないこと | 推奨規模 |
| --- | --- | --- | --- |
| ENV-001 | 5.1、5.3、3 | clean install、aarch64、明示CUDA、BF16、silent fallbackなし | R2 |
| DATA-001 | 4.3、4.1、6.1 | transitionの欠落/重複なしと、carry導入後の供給速度 | R1 + R2 |
| MODEL-001 | 6.1、6.2、5.2、7 | baseline invariantを守り、reference modelのstep time/memoryを記録 | R1 + R2 |
| EXP-001 | 8.1、8.3、7 | negative resultと性能条件を省略できないhandoff | R0 |
| TOK-001 | 4.2、4.1、5.4、8.2 | 日本語fallback、offline pin、vocab由来のmodel/optimizerコスト | R1 + R2 |
| DATA-002 | 4.4、8.2、4.1 | split/checksum確認をhot pathへ毎sample入れていない | R1 |
| CFG-001 | 1、2、7.1、8.1 | real profileが実sourceをcomposeし、debug configと二重化しない | R1 |
| REP-001 | 8.1、3、6.1 | seed範囲、run identity、deterministic設定の速度trade-off | R1 + 必要ならR2 |
| DATA-003 | 4.1、4.3、9.1 | shuffle buffer/cursorのmemory・size・供給速度とexact suffix | R2 |
| LOOP-001 | 6全体、3、9.2 | counters、token-weighted metric、event cadence、毎step同期 | R2 |
| CKPT-001 | 9.1、5.4、6.3 | atomic saveの実サイズpauseと、resume後trajectory/cursor | R2 |
| CI-001 | 1、7.3、8.1 | network/credentialなし、canonical local commandとの一致 | R0 + R1 |
| OPT-001 | 5.2、6.1、6.2、3 | BF16実kernel、accumulation意味、clip率、optimizer step throughput | R2 + R3 |
| GEN-001 | 7.2、8.2、9.1 | training pathへchat/SFT分岐を持ち込まずcheckpointから再構築 | R1 |
| WB-001 | 9.2、6.3、5.4 | logging/watch/upload off/onのoverheadとquota障害分離 | R2 |
| GATE-001 | 6、8、9.1 | memorizationと明記し、resumeとsamplingまで一つのchainで確認 | R2 |
| DATA-004 | 4全体、5.3、5.4、8.2 | real sourceのcold/warm throughput、長文tail、disk headroom | R2 + R3 |
| VAL-001 | 6.1、8.2、6.3 | standalone/training-time parity、token budget、学習停止時間 | R2 |
| BENCH-001 | 8.2、8.3、9.2 | reserved test guard、contamination、decoding/scorer identity | R1 |
| DGX-001 | 3、4.1、5全体、6.4 | model-only/loader-only/end-to-end分解、thermalを含む反復計測 | R3 |
| OPS-001 | 9.3、8.1、5.4 | preflightが破壊的でなく、失敗runとretryの証拠を残す | R2 end-to-end |
| RUN-001 | 9.3、4〜9の該当部 | pilotから直接導いた時間/token/storage計画と停止条件 | R4 |
| HUMAN-001 | 8.2、8.3 | blind漏洩なし、prompt/output/scoreをtrainingへ戻さない | R1 |

### 性能最適化チケットを後で追加するとき

`ROADMAP.md` のDeferred方針に加えて、最低限次をチケット本文に書く。

- 測定されたbottleneckと、それを示すtraceまたは内訳
- 基準commit、基準config、基準結果
- objective、対象transition、数値結果に対するcorrectness tolerance
- 変更後に期待するmetricと最小有意差
- 測定区間、warmup、反復回数、cache、profiler/logging条件
- 速度、memory、品質、複雑性のtrade-off
- 改善しなかった場合に削除してreferenceへ戻すrollback条件

## 12. ボトルネック診断の順番

### trainingが遅い

1. target tokens/sとstep timeをbaselineと比較する。
2. synthetic device batchのmodel-onlyを測る。
3. 実loader-onlyを測る。
4. end-to-endのdata waitとGPU gapを見る。
5. model-onlyだけ遅ければkernel、dtype、shape、同期、optimizerを見る。
6. loader-onlyだけ遅ければsource、tokenizer、packing、IPC、cache/networkを見る。
7. 単体は速くend-to-endだけ遅ければcopy、同期、logging、queue、resource競合を見る。
8. 短時間は速くpilotで落ちるならthermal、clock、swap、cache eviction、network tailを見る。

### GPU utilizationが低い

1. まず実CUDA processとCUDA kernelを確認する。
2. tokens/sが十分なら、小モデルや短kernelのためutilization平均だけが低い可能性を考える。
3. GPU timelineに空白があれば、同時刻の`next(loader)`、CPU blocked、`.item()`、loggingを見る。
4. 空白がなくても遅ければ、kernel launch-bound、memory-bound、非Tensor Core dtype/shapeを調べる。
5. batch/sequenceを増やすscale curveでparallelism不足かを確認する。

### memoryが増える

1. PyTorch allocated、reserved、process RSS、system available、swapを分ける。
2. warmup/autotuneによる一度きりの増加と、stepごとの単調増加を分ける。
3. retained graph、logging/watch、metrics list、prefetch queue、tokenizer long doc、file cacheを確認する。
4. checkpoint/validation時だけのpeakか、通常stepでも戻らないかを見る。
5. DGX SparkではGPU memory表示だけでなくUMA全体を見る。

### lossまたは品質が悪化した

1. tokenizer、data、target token、mask、reductionが同じか確認する。
2. effective batch、LR schedule、optimizer step数、precision、seedを確認する。
3. train loss、held-out NLL、generation/benchmarkのどこから悪化したか分ける。
4. throughput改善のためにtargetを落としたりpaddingを数えなくなったりしていないか確認する。
5. 数値差なのか、実験定義が変わって比較不能なのかを先に判断する。

## 13. レビュー記録テンプレート

必要な項目だけ残してチケットまたはPRへ貼る。

```markdown
## ML system check

- Ticket / hypothesis:
- Verdict: PASS / PASS WITH NOTE / FAIL
- Baseline commit or run:
- Candidate commit or run:
- Selected sections: 例 4.1, 5.2, 6.3, 7.1
- Why the other major sections are N/A:

### Conditions
- Hydra command and resolved config:
- Data/tokenizer/model fingerprints:
- Hardware / OS / driver / CUDA / PyTorch:
- Precision / sequence / micro batch / accumulation:
- Warmup / measured steps / repetitions / cache state:
- W&B / profiler / checkpoint / validation settings:

### Result
- Intended behavior:
- target tokens/s, baseline -> candidate:
- step median / p95, baseline -> candidate:
- data wait and GPU gap:
- GPU clock / power / temperature:
- system available memory / swap / PyTorch peak:
- loss / gradient / LR health:
- checkpoint/eval/logging overhead:

### Engineering judgment
- Objective or data semantics changed?:
- New coupling, duplicate path, or config source?:
- Next model/data experiment remains localized?:
- Known trade-off and why acceptable:
- Unresolved risk and next action:
- Evidence paths / run URLs / trace paths:
```

## 14. 観測ツールの使い分け

チケットで必要なものだけ使う。すべてを常設したり、すべてのrunをprofileしたりしない。

- `nvidia-smi` / `nvidia-smi dmon`: compute process、GPU active、power、clock、temperatureの低頻度時系列。DGX Sparkではmemory表示の制約に注意する。
- `free -h`、`vmstat 1`: UMA全体のavailable memory、swap、run queue、I/O waitを見る。
- `pidstat`、`iostat`: 利用可能なら、学習processのCPU/RSS/I/OとNVMe待ちを分ける。
- PyTorch Profiler: 数十stepだけCPU/CUDA activityを取り、operator、copy、同期、shape、memoryを調べる。warmup/active scheduleを使う。
- Nsight Systems: GPU idle gapとCPU blocked、CUDA API、kernel、memcpy、NVTX区間を同じtimelineで見る。PyTorch Profilerで原因が足りないときに使う。
- W&B/local JSON: 長いpilot/runの低頻度傾向を見る。詳細traceの代わりにしない。
- plain wall-clock timer: loader-only、model-only、checkpoint、validationなど境界の明確な処理を反復測定する。

計測用のshape記録、stack trace、CUDA memory trace、細かいGPU metricsは無視できないoverheadを持ち得る。
通常runとprofile runを分け、profileを有効にした結果を通常throughputとして報告しない。

## 15. 参考にした一次・公式資料

- [DGX Spark Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html): ARM64、128 GB unified memory、powerとhardware仕様。
- [DGX Spark User Guide / Known Issues](https://docs.nvidia.com/dgx/dgx-spark/index.html): UMAでのmemory reportingと`nvidia-smi`の注意。
- [DGX Spark Porting Guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/index.html): ARM/UMA、profiling、DGX Sparkへの移植・最適化。
- [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html): utilization、power、clock、temperature、`dmon`の定義。
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/): CPU/CUDA timeline、GPU metrics、UMA/page fault観測。
- [Nsight Systems Analysis Guide](https://docs.nvidia.com/nsight-systems/2025.5/AnalysisGuide/index.html): GPU starvation/low utilizationとCPU blocked状態の診断。
- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler): CPU/CUDA activity、warmup/active schedule、trace取得。
- [PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html): iterable dataset、worker、prefetch、pin memoryの挙動。
- [PyTorch Data Loading Optimization](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html): pinning、non-blocking transfer、data pipeline計測。
- [PyTorch Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html): seed、platform差、deterministic設定と性能のtrade-off。
- [PyTorch Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/accelerator/amp.html): autocast、低precision、gradient scaling。
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/index.html): math/latency/memory bound、Tensor Core、shapeとdata movement。

外部のversion、quota、hardware/software制約は変更される。チケット開始時には現在の公式資料を
再確認し、この文書中の数値やコマンドを永久的な定数として扱わない。
