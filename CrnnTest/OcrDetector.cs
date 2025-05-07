using Emgu.CV;
using Emgu.CV.CvEnum;
using System;
using System.Drawing;

namespace CrnnTest
{
    public class OcrDetector
    {
        private string modelPath;

        public OcrDetector(string modelPath)
        {
            this.modelPath = modelPath;
        }

        public Mat CropAndResize(Mat src, Rectangle roi, int width, int height)
        {
            if (src == null || src.IsEmpty)
            {
                Console.WriteLine("입력 이미지가 null이거나 비어 있음");
                return new Mat(); // 빈 Mat 반환
            }

            Rectangle safeRoi = ClampRoi(roi, src.Size);

            if (safeRoi.Width <= 0 || safeRoi.Height <= 0)
            {
                Console.WriteLine($"잘못된 ROI: {safeRoi}");
                return new Mat(src.Size, DepthType.Cv8U, 3); // 대체용
            }

            if (safeRoi.X + safeRoi.Width > src.Width || safeRoi.Y + safeRoi.Height > src.Height)
            {
                Console.WriteLine($"ROI 초과: {safeRoi} / Image Size: {src.Width}x{src.Height}");
                return new Mat(src.Size, DepthType.Cv8U, 3);
            }

            Mat cropped = new Mat(src, safeRoi);
            Mat resized = new Mat();
            CvInvoke.Resize(cropped, resized, new Size(width, height));
            return resized;
        }

        private Rectangle ClampRoi(Rectangle roi, Size imgSize)
        {
            int x = Math.Max(0, roi.X);
            int y = Math.Max(0, roi.Y);

            int width = roi.Width;
            int height = roi.Height;

            if (x + width > imgSize.Width)
                width = imgSize.Width - x;
            if (y + height > imgSize.Height)
                height = imgSize.Height - y;

            width = Math.Max(1, width);
            height = Math.Max(1, height);

            return new Rectangle(x, y, width, height);
        }
    }
}
