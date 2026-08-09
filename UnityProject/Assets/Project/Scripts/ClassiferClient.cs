using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class ClassiferClient : MonoBehaviour
{
    [SerializeField] private SentimentClassifier _classifier;
    [SerializeField] private Button _button;
    [SerializeField] private TMP_InputField _inputField;
    [SerializeField] private TMP_Text _resultText;
    
    private void Start()
    {
        _button.onClick.AddListener(() =>
        {
            // inputをclassifierする実装
            var (label, score) = _classifier.Predict(_inputField.text);
            _resultText.text = $"{label} {score}";
        });
    }
}
