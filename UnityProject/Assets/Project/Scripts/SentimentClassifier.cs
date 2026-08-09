using Newtonsoft.Json;
using System.Collections.Generic;
using UnityEngine;
using Unity.InferenceEngine;

public class SentimentClassifier : MonoBehaviour
{
    [SerializeField] private ModelAsset _modelAsset;
    [SerializeField] private TextAsset _id2labelJson;
    [SerializeField] private TextAsset _tokenizerJson;

    private Worker _worker;
    private Dictionary<int, string> _id2label;
    private ITokenizer _tokenizer;
    private const int MaxLength = 192;

    private void Awake()
    {
        Model model = ModelLoader.Load(_modelAsset);
        _worker = new Worker(model, BackendType.GPUCompute);
        _id2label = ParseId2Label(_id2labelJson.text);
        _tokenizer = new UnigramTokenizer(_tokenizerJson.text);
    }

    private Dictionary<int, string> ParseId2Label(string json)
    {
        return JsonConvert.DeserializeObject<Dictionary<int, string>>(json);
    }

    public (string label, float score) Predict(string text)
    {
        (int[] ids, int[] mask) = _tokenizer.Encode(text, MaxLength);

        // 推論
        // テンソルの形を決め、データを設定する
        var shape = new TensorShape(1, ids.Length);
        using var idsTensor = new Tensor<int>(shape, ids);
        using var maskTensor = new Tensor<int>(shape, mask);

        // ワーカーに入力を設定
        // "input_ids" などは ONNX モデルの情報から判断できる。（インスペクタで見れる）
        // モデルの入力として "input_ids" と "attention_mask" を受け取る。
        _worker.SetInput("input_ids", idsTensor);
        _worker.SetInput("attention_mask", maskTensor);
        _worker.Schedule();

        // GPU 上で計算した結果を取り出す。取り出す名前は、上記入力と同様、
        // モデルから確認できる出力名である "logits" を指定する。
        using var logitsGpu = _worker.PeekOutput("logits") as Tensor<float>;
        if (logitsGpu == null)
        {
            return ("None", 0);
        }

        using var logits = logitsGpu.ReadbackAndClone();
        float[] scores = Softmax(logits.DownloadToArray());

        // argmax → 日本語ラベル
        int pred = ArgMax(scores);
        return (_id2label[pred], scores[pred]);
    }

    private void OnDestroy()
    {
        _worker?.Dispose();
    }

    private static float[] Softmax(float[] logits)
    {
        float max = logits[0];
        foreach (var logit in logits)
        {
            if (logit > max)
            {
                max = logit;
            }
        }

        float sum = 0f;
        float[] e = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            // 最大値を引くことで、全体を全体値分シフトしている。
            // これによってオーバーフローなどの問題を防ぐことができる。
            e[i] = Mathf.Exp(logits[i] - max);
            sum += e[i];
        }

        for (int i = 0; i < e.Length; i++)
        {
            e[i] /= sum;
        }

        return e;
    }

    static int ArgMax(float[] a)
    {
        int b = 0;
        for (int i = 1; i < a.Length; i++)
            if (a[i] > a[b])
                b = i;
        return b;
    }
}