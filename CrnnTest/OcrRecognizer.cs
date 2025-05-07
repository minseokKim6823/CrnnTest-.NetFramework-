using Emgu.CV;


namespace CrnnTest
{
    public class OcrRecognizer
    {
        private string modelPath;

        public OcrRecognizer(string modelPath)
        {
            this.modelPath = modelPath;
        }

        public string Run(Mat input)
        {
            // 실제 ONNX inference 로직은 생략, 예시로 반환
            return "SERIAL or AMOUNT result";
        }
    }
}