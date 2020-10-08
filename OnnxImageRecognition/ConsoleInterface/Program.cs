using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ImageRecognizer;

namespace ConsoleInterface
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("Type the full path to the folder, and image recognition will begin:");
            string path = Console.ReadLine();
            var traversing = MnistRecognizer.TraverseDirectory(path);

            
            while (!recognitionFinishedMarkers.Contains(traversing.Status))
            {
                await MnistRecognizer.NewResults.WaitAsync();
                await MnistRecognizer.WritePermission.WaitAsync();
                foreach (var entry in MnistRecognizer.ResultsQueue)
                {
                    Console.WriteLine($"Results for {entry.ImagePath}:");

                    foreach (var output in entry.ModelOutput)
                    {
                        Console.WriteLine($"{output.Label} with confidence {output.Confidence}");
                    }
                }
                MnistRecognizer.ResultsQueue.Clear();
                MnistRecognizer.WritePermission.Release();
                await Task.Delay(100);
            }

            foreach (var entry in MnistRecognizer.ResultsQueue)
            {
                Console.WriteLine($"Results for {entry.ImagePath}:");

                foreach (var output in entry.ModelOutput)
                {
                    Console.WriteLine($"{output.Label} with confidence {output.Confidence}");
                }
            }
        }

        static readonly TaskStatus[] recognitionFinishedMarkers = {
            TaskStatus.Faulted,
            TaskStatus.Canceled,
            TaskStatus.RanToCompletion,
        };
    }
}
