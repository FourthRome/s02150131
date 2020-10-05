// Пример на основе https://github.com/microsoft/onnxruntime/tree/master/csharp/sample/Microsoft.ML.OnnxRuntime.ResNet50v2Sample

using System;
using SixLabors.ImageSharp; // Из одноимённого пакета NuGet
using SixLabors.ImageSharp.PixelFormats;
using System.Linq;
using SixLabors.ImageSharp.Processing;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;

namespace ImageRecognizer
{
    public class MnistRecognizer
    {
        public static void Recognize(string path)
        {
            Image<Rgb24> image = null;
            try
            {
                image = Image.Load<Rgb24>(path);
            } catch(Exception e)
            {
                // TODO: proper logging
                System.Diagnostics.Debug.WriteLine($"[FILE ERROR] MnistRecognizer.Recognize: Could not read image \"{path}\": \n{e.Message}");
                return;
            }

            // These come from the model's requirements
            const int TargetWidth = 28;
            const int TargetHeight = 28;

            // Изменяем размер картинки до 28 x 28
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(TargetWidth, TargetHeight),
                    Mode = ResizeMode.Crop // Сохраняем пропорции обрезая лишнее
                });

                x.Grayscale();
            });

            // Перевод пикселов в тензор и нормализация
            var input = new DenseTensor<float>(new[] { 1, 1, TargetHeight, TargetWidth });
            for (int y = 0; y < TargetHeight; y++)
            {
                Span<Rgb24> pixelSpan = image.GetPixelRowSpan(y);
                for (int x = 0; x < TargetWidth; x++)
                {
                    input[0, 0, y, x] = pixelSpan[x].R / 255f;
                }
            }

            // Подготавливаем входные данные нейросети. Имя Input3 задано в файле модели
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("Input3", input)
            };

            // Вычисляем предсказание нейросетью
            using var session = new InferenceSession("mnist-8.onnx");
            Console.WriteLine("Predicting contents of image...");
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

            // Получаем 1000 выходов и считаем для них softmax
            var output = results.First().AsEnumerable<float>().ToArray();
            var sum = output.Sum(x => (float)Math.Exp(x));
            var softmax = output.Select(x => (float)Math.Exp(x) / sum);

            // Выдаем 10 наиболее вероятных результатов на экран
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
