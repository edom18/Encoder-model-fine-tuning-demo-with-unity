using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json.Linq;
using UnityEngine;

public class UnigramTokenizer : ITokenizer
{
    private readonly float[] _scores; // id → スコア（対数確率）
    private readonly Dictionary<string, int> _pieceToId; // ピース → id
    private readonly int _unkId;
    private readonly int _bosId;
    private readonly int _eosId;
    private readonly int _maxPieceLen;
    private const char Meta = '▁';

    public UnigramTokenizer(string tokenizerJson)
    {
        var model = JObject.Parse(tokenizerJson)["model"];
        _unkId = (int)model["unk_id"];
        var vocab = (JArray)model["vocab"];
        int n = vocab.Count;
        _scores = new float[n];
        _pieceToId = new Dictionary<string, int>(n);
        for (int i = 0; i < n; i++)
        {
            var e = (JArray)vocab[i];
            string piece = (string)e[0];
            _scores[i] = (float)e[1];
            _pieceToId[piece] = i;
            if (piece.Length > _maxPieceLen)
            {
                _maxPieceLen = piece.Length;
            }
        }

        _bosId = _pieceToId["<s>"];
        _eosId = _pieceToId["</s>"];
    }
    
    public (int[] ids, int[] mask) Encode(string text, int maxLength)
    {
        string s = text.Replace(' ', Meta); // 空白を '_' に変換
        List<int> body = Viterbi(s);        // 最尤分割 → id 列
        int budget = _maxPieceLen - 2;
        if (body.Count > budget)
        {
            body = body.GetRange(0, budget);
        }

        var ids = new List<int>(body.Count + 2)
        {
            _bosId,
        };
        ids.AddRange(body);
        ids.Add(_eosId);

        int[] idArr = ids.ToArray();
        int[] mask = new int[idArr.Length];
        for (int i = 0; i < mask.Length; i++)
        {
            mask[i] = 1; // 全部 1（パディングなし）
        }

        return (idArr, mask);
    }

    private List<int> Viterbi(string s)
    {
        // -- 準備: 文字列を「コードポイント境界」の配列にする --
        // off[k] = k 番目の文字が始まる UTF-16 インデックス
        // 絵文字など 2 単位で 1 文字のものを「1つ」として扱い、途中で割らないため。
        // 文字列を code point 境界の配列に（絵文字/サロゲートも1要素扱い）
        var off = new List<int>();
        for (int i = 0; i < s.Length; i += char.IsHighSurrogate(s[i]) ? 2 : 1)
        {
            off.Add(i); // 各文字の開始位置を記録
        }

        off.Add(s.Length);       // 末尾番兵（最後の区切り = 文字列長）
        int len = off.Count - 1; // 文字数（コードポイント数）
        
        // 位置 b..e の部分文字列を取り出すヘルパ（境界経由なので絵文字が割れない）
        string Piece(int b, int e) => s.Substring(off[b], off[e] - off[b]);

        // -- DP テーブル --
        // best[i] = 「先頭から i 文字目までを語彙ピースで区切ったときの
        //             合計スコア（対数確率）の最大値」。大きいほど尤もらしい
        var best = new double[len + 1]; // best[i] = 位置 i までの最良スコア
        // back[i] = その最良分割で「最後のピースが始まる位置」。復元用の後ろ向きポインタ。
        var back = new int[len + 1];    // back[i] = 最後のピースの開始位置
        
        // 初期化: まだ到達していない印として全部 -∞
        for (int i = 1; i <= len; i++)
        {
            best[i] = double.NegativeInfinity;
        }

        best[0] = 0; // 「0文字目まで」= 空。スコア 0（= log 1）。出発点。

        // -- 前向き DP（左 → 右）: 各終端 end の best を確定していく --
        for (int end = 1; end <= len; end++)
        {
            // end で終わるピースの「開始位置」を全部試す。
            // 語彙の最良ピース長 maxPieceLen より手前は見ても無駄なので打ち切る
            int from = Mathf.Max(0, end - _maxPieceLen);
            for (int begin = from; begin < end; begin++) // end で終わる全ピース
            {
                // begin にまだ到達できていない（-∞）なら、そこ起点の分割は作れない → 飛ばす。
                if (double.IsNegativeInfinity(best[begin])) continue;
                
                // s[begin..end] が語彙にあるピースか？（= この辺がラティスに存在するか）
                if (_pieceToId.TryGetValue(Piece(begin, end), out int id))
                {
                    // 「begin までの最良」+「このピースの対数確率」= end までの候補スコア
                    // ユ二グラム = 独立なので、遷移コストなしで単純に足すだけ。
                    double cand = best[begin] + _scores[id];
                    
                    // これまでの best[end] より良ければ更新し、後ろ向きポインタも張り替える
                    if (cand > best[end])
                    {
                        best[end] = cand;
                        back[end] = begin;
                    }
                }
            }

            // どのピースも end に届かなかった（= 語彙外の 1 文字）場合の保険
            // 直前の 1 文字だけを「バイト分解行き」の区間として無理やりつなぐ。
            // 罰 -10 は「語彙ピースがあるならそちらを優先」させるための相対値。
            if (double.IsNegativeInfinity(best[end])) // どのピースでも届かない → 1 文字だけ fallback
            {
                best[end] = best[end - 1] - 10.0; // 罰（順位付けのためだけ）
                back[end] = end - 1;
            }
        }

        // -- 後ろ向き復元（右 → 左）: back を遡って区間列を取り出す --
        var outIds = new List<int>();
        var segs = new List<(int b, int e)>();
        for (int pos = len; pos > 0; pos = back[pos])
        {
            // （最後のピースの開始, 終了）を末尾から集める
            segs.Add((back[pos], pos));
        }

        // 末尾から集めたので、文頭 → 文末の順に直す
        segs.Reverse();
        
        // 各区間を id に変換。語彙にあればその id、なければ byte_fallback。
        foreach (var (b, e) in segs)
        {
            string sub = Piece(b, e);
            if (_pieceToId.TryGetValue(sub, out int id))
            {
                outIds.Add(id);
            }
            else
            {
                EmitBytes(sub, outIds);
            }
        }

        return outIds;
    }

    private void EmitBytes(string sub, List<int> outIds)
    {
        foreach (byte b in Encoding.UTF8.GetBytes(sub))
        {
            outIds.Add(_pieceToId.GetValueOrDefault($"<0x{b:X2}>", _unkId));
        }
    }
}
