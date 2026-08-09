using UnityEngine;
using Unity.InferenceEngine;

public class SentisSmokeTest : MonoBehaviour
{
    [SerializeField] private ModelAsset _modelAsset;
    
    private void Start()
    {
        int[] inputIds = new int[] { 1, 1707, 7023, 6151, 894, 328, 2 };
        int[] attentionMask = new int[] { 1, 1, 1, 1, 1, 1, 1 };
        int seqLen = inputIds.Length;

        Model model = ModelLoader.Load(_modelAsset);
        Worker worker = new Worker(model, BackendType.GPUCompute);

        TensorShape shape = new TensorShape(1, seqLen);
        using Tensor<int> idsTensor = new Tensor<int>(shape, inputIds); // int32
        using Tensor<int> maskTensor = new Tensor<int>(shape, attentionMask);

        worker.SetInput("input_ids", idsTensor);
        worker.SetInput("attention_mask", maskTensor);
        worker.Schedule();
        
        // 出力は backend 上にある → CPU へ読み戻してから読む
        using Tensor<float> logitsGpu = worker.PeekOutput("logits") as Tensor<float>;
        if (logitsGpu == null)
        {
            Debug.LogError("Failed to get logits from the worker.");
            return;
        }
        using Tensor<float> logits = logitsGpu.ReadbackAndClone();
        float[] scores = logits.DownloadToArray();

        int pred = ArgMax(scores);
        Debug.Log($"pred={pred} logits[0]={scores[0]:F2} (期待 ~15.40, pred=0=喜び)");
        
        worker.Dispose();
    }

    private int ArgMax(float[] a)
    {
        int best = 0;
        for (int i = 1; i < a.Length; i++)
        {
            if (a[i] > a[best])
            {
                best = i;
            }
        }

        return best;
    }
}
