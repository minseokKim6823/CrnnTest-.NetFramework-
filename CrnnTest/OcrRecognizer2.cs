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
    public class OcrRecognizer2
    {
        private InferenceSession session;
        private Dictionary<int, char> idxToChar;

        public OcrRecognizer2(string modelPath)
        {
            session = new InferenceSession(modelPath);
            idxToChar = LoadCharsetFromFile("micr_dict.txt");
        }

        public string Run(Mat x)
        {
            if (x == null || x.IsEmpty)
                return "";

            // 1. 전처리: Grayscale + Resize(48x320) + Normalize
            Mat gray = new Mat();
            CvInvoke.CvtColor(x, gray, ColorConversion.Bgr2Gray);
            CvInvoke.Resize(gray, gray, new Size(320, 48));
            gray.ConvertTo(gray, DepthType.Cv32F, 1.0 / 255);

            // 2. Tensor 준비 (1, 1, 48, 320)
            var img = gray.ToImage<Gray, float>();
            var data = img.Data;
            float[] inputData = new float[1 * 3 * 48 * 320];
            int idx = 0;
            for (int i = 0; i < 48; i++)
                for (int j = 0; j < 320; j++)
                    inputData[idx++] = data[i, j, 0];

            var tensor = new DenseTensor<float>(inputData, new[] { 1, 3, 48, 320 });
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("x", tensor) };
            var results = session.Run(inputs);
            var output = results.First().AsTensor<float>();  // shape: (1, T, C)

            // 3. CTC 디코딩 (Greedy)
            int T = output.Dimensions[1];
            int C = output.Dimensions[2];
            float[] flat = output.ToArray();

            List<int> bestPath = new List<int>();
            for (int t = 0; t < T; t++)
            {
                int maxIndex = 0;
                float maxProb = flat[t * C + 0];
                for (int c = 1; c < C; c++)
                {
                    if (flat[t * C + c] > maxProb)
                    {
                        maxProb = flat[t * C + c];
                        maxIndex = c;
                    }
                }
                bestPath.Add(maxIndex);
            }

            // 4. Post-process: CTC blank 제거 + 반복 제거
            string result = "";
            int prev = -1;
            foreach (int i in bestPath)
            {
                if (i != 0 && i != prev)
                {
                    if (idxToChar.ContainsKey(i))
                        result += idxToChar[i];
                }
                prev = i;
            }

            return result;
        }

        private Dictionary<int, char> LoadCharsetFromFile(string dictPath)
        {
            var lines = System.IO.File.ReadAllLines(dictPath);
            var map = new Dictionary<int, char>();

            for (int i = 0; i < lines.Length; i++)
            {
                string line = lines[i].Trim();
                if (!string.IsNullOrEmpty(line) && line.Length == 1)
                {
                    map[i + 1] = line[0]; // CTC blank = 0
                }
            }

            return map;
        }
    }
}
