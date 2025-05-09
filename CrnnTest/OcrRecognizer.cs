using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace CrnnTest
{
    public class OcrRecognizer
    {
        private InferenceSession session;
        private Dictionary<int, char> idxToChar;

        public OcrRecognizer(string modelPath)
        {
            session = new InferenceSession(modelPath);

            // 인덱스 -> 문자 매핑 (CTC 모델용)
            string charset = "0123456789\\,.* ";
            idxToChar = charset.Select((c, i) => new { i, c }).ToDictionary(x => x.i + 1, x => x.c); // blank=0
        }

        public string Run(Mat input)
        {
            if (input == null || input.IsEmpty)
            {
                throw new ArgumentException("입력 이미지가 null이거나 비어 있습니다.");
            }

            Mat gray = new Mat();
            CvInvoke.Resize(input, input, new Size(320, 48));
            input.ConvertTo(input, DepthType.Cv32F, 1.0 / 255);
            Image<Bgr, float> img = input.ToImage<Bgr, float>();

            float[] inputData = new float[3 * 48 * 320];
            int idx = 0;
            for (int c = 0; c < 3; c++)
                for (int i = 0; i < 48; i++)
                    for (int j = 0; j < 320; j++)
                        inputData[idx++] = img.Data[i, j, c];

            var tensor = new DenseTensor<float>(inputData, new[] { 1, 3, 48, 320 });
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("x", tensor) };

            // 3. ONNX Inference
            var results = session.Run(inputs);
            var output = results.First().AsTensor<float>(); // (1, T, C)
            int T = output.Dimensions[1], C = output.Dimensions[2];

            float[] flat = output.ToArray();
            List<int> bestPath = new List<int>();
            for (int t = 0; t < T; t++)
            {
                int maxIndex = 0;
                float maxProb = flat[t * C + 0];
                for (int c = 1; c < C; c++)
                {
                    float prob = flat[t * C + c];
                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxIndex = c;
                    }
                }
                bestPath.Add(maxIndex);
            }

            // 5. CTC Post-process (remove duplicates and blank=0)
            List<char> decoded = new List<char>();
            int prev = -1;
            foreach (int curr in bestPath)
            {
                if (curr != 0 && curr != prev && idxToChar.ContainsKey(curr))
                    decoded.Add(idxToChar[curr]);
                prev = curr;
            }

            return new string(decoded.ToArray());
        }
    }
}