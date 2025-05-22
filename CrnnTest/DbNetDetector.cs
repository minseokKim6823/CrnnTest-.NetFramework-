using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using static System.Net.Mime.MediaTypeNames;

namespace CrnnTest
{
    public class DbNetDetector
    {
        private InferenceSession session;
        private int boxSizeThresh = 4;

        public DbNetDetector(string modelPath)
        {
            session = new InferenceSession(modelPath);
        }

       

        public List<Rectangle> Detect(Mat image)
        {
            int origW = image.Width;
            int origH = image.Height;
            int inputW = 480;
            int inputH = 320;

            Rectangle amountRoi = new Rectangle(
               (int)(origW * 0.2),
               (int)(origH * 0.25),
               (int)(origW * 0.6),
               (int)(origH * 0.05)
            );

            // DBNet 모델은 RGB 입력을 받음
            Mat rgbImage = new Mat();
            CvInvoke.CvtColor(image, rgbImage, ColorConversion.Bgr2Rgb);

            // 2. Letterbox (비율 유지) 리사이징 + 패딩
            float scale = Math.Min((float)inputW / origW, (float)inputH / origH);
            int resizedW = (int)(origW * scale);
            int resizedH = (int)(origH * scale);

            Mat resized = new Mat();
            CvInvoke.Resize(rgbImage, resized, new Size(resizedW, resizedH));
            resized.ConvertTo(resized, DepthType.Cv32F, 1.0 / 255);  // 정규화

            Image<Bgr, float> resizedImg = resized.ToImage<Bgr, float>();
            Image<Bgr, float> paddedImg = new Image<Bgr, float>(inputW, inputH);
            paddedImg.SetZero();

            Rectangle roi = new Rectangle(
                (inputW - resizedW) / 2,
                (inputH - resizedH) / 2,
                resizedW, resizedH
            );
            if (resizedImg.Size == roi.Size)
            {
                paddedImg.ROI = roi;
                resizedImg.CopyTo(paddedImg);
                paddedImg.ROI = Rectangle.Empty;
            }
            else
            {
                resizedImg.Mat.CopyTo(new Mat(paddedImg.Mat, roi));
            }
            // 3. ONNX 입력 준비
            var data = paddedImg.Data;
            float[] inputData = new float[3 * inputH * inputW];
            int idx = 0;
            for (int c = 0; c < 3; c++)
                for (int i = 0; i < inputH; i++)
                    for (int j = 0; j < inputW; j++)
                        inputData[idx++] = data[i, j, c];

            var tensor = new DenseTensor<float>(inputData, new[] { 1, 3, inputH, inputW });
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("x", tensor) };
            var results = session.Run(inputs);
            var output = results.First().AsTensor<float>();

            int rows = output.Dimensions[2];
            int cols = output.Dimensions[3];
            float[] flat = output.ToArray();

            Image<Gray, float> maskImage = new Image<Gray, float>(cols, rows);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    maskImage.Data[i, j, 0] = flat[i * cols + j];

            Mat mask = maskImage.Mat;
            Mat binMask = new Mat();
            CvInvoke.Threshold(mask, binMask, 0.4, 255, ThresholdType.Binary);
            binMask.ConvertTo(binMask, DepthType.Cv8U);

            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            CvInvoke.FindContours(binMask, contours, null, RetrType.External, ChainApproxMethod.ChainApproxSimple);

            List<Rectangle> boxes = new List<Rectangle>();
            for (int i = 0; i < contours.Size; i++)
            {
                Rectangle rect = CvInvoke.BoundingRectangle(contours[i]);
                if (rect.Width > boxSizeThresh && rect.Height > boxSizeThresh)
                {
                    Rectangle boxInOrig = new Rectangle(
                        (int)((rect.X - roi.X) / scale),
                        (int)((rect.Y - roi.Y) / scale),
                        (int)(rect.Width / scale),
                        (int)(rect.Height / scale)
                    );

                    // amount 영역에 포함되면 padding 추가
                    int padding = amountRoi.IntersectsWith(boxInOrig) ? 10 : 6;

                    Rectangle corrected = new Rectangle(
                        Math.Max(0, boxInOrig.X - padding),
                        Math.Max(0, boxInOrig.Y - padding),
                        Math.Min(origW - boxInOrig.X, boxInOrig.Width + 2 * padding),
                        Math.Min(origH - boxInOrig.Y, boxInOrig.Height + 2 * padding)
                    );

                    if (corrected.X >= 0 && corrected.Y >= 0 && corrected.Right <= origW && corrected.Bottom <= origH)
                        boxes.Add(corrected);
                }
            }


            return boxes;
        }
    }
}