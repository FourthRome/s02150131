using System;
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
            await MnistRecognizer.TraverseDirectory(path);

            foreach (var entry in MnistRecognizer.ResultsQueue)
            {
                Console.WriteLine($"Results for {entry.ImagePath}:");
                
                foreach (var output in entry.ModelOutput)
                {
                    Console.WriteLine($"{output.Label} with confidence {output.Confidence}");
                }
            }
        }
    }
}
