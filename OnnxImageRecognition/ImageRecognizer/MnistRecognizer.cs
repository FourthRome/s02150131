// Пример на основе https://github.com/microsoft/onnxruntime/tree/master/csharp/sample/Microsoft.ML.OnnxRuntime.ResNet50v2Sample

using System;
using SixLabors.ImageSharp; // Из одноимённого пакета NuGet
using SixLabors.ImageSharp.PixelFormats;
using System.Linq;
using SixLabors.ImageSharp.Processing;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using System.IO;

namespace ImageRecognizer
{
    public class MnistRecognizer
    {
        public static void TraverseDirectory(string path)
        {
            System.IO.DirectoryInfo dir = new System.IO.DirectoryInfo(path);
            if (dir.Exists)
            {
                foreach (FileInfo fi in dir.GetFiles())
                {
                    Recognize(fi.FullName);
                }
            } else
            {
                System.Diagnostics.Trace.WriteLine($"[INCORRECT PATH] MnistRecognizer.TraverseDirectory: Could not open {path}; the directory might not exist.");
            }
        }

        public static void Recognize(string path)
        {
            Image<Rgb24> image = null;
            try
            {
                image = Image.Load<Rgb24>(path);
            } catch(Exception e)
            {
                // TODO: proper logging
                System.Diagnostics.Trace.WriteLine($"[FILE ERROR] MnistRecognizer.Recognize: Could not read image \"{path}\": \n{e.Message}");
                return;
            }

            // These come from the model's requirements
            const int TargetWidth = 28;
            const int TargetHeight = 28;

            // Changing image size and making it grayscale
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(TargetWidth, TargetHeight),
                    Mode = ResizeMode.Crop // Сохраняем пропорции обрезая лишнее
                });
                x.Grayscale();
            });

            // Converting to a tensor and normalization
            var input = new DenseTensor<float>(new[] { 1, 1, TargetHeight, TargetWidth });
            for (int y = 0; y < TargetHeight; y++)
            {
                Span<Rgb24> pixelSpan = image.GetPixelRowSpan(y);
                for (int x = 0; x < TargetWidth; x++)
                {
                    input[0, 0, y, x] = pixelSpan[x].R / 255f;
                }
            }

            // Preparing neural net's inputs. 'Input3' name is used in the model itself
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("Input3", input)
            };

            // Inference
            using var session = new InferenceSession("mnist-8.onnx");
            Console.WriteLine("Predicting contents of image...");
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

            // Applying softmax
            // TODO: get into details and see if this is necessary for MNIST
            var output = results.First().AsEnumerable<float>().ToArray();
            var sum = output.Sum(x => (float)Math.Exp(x));
            var softmax = output.Select(x => (float)Math.Exp(x) / sum);

            // Printing results
            foreach (var p in softmax
                .Select((x, i) => new { Label = classLabels[i], Confidence = x })
                .OrderByDescending(x => x.Confidence)
                .Take(10))
                Console.WriteLine($"{p.Label} with confidence {p.Confidence}");
        }

        static readonly string[] classLabels = new[]
        {
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        };
    }
}
