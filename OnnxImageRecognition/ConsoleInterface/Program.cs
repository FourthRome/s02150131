using System;
using ImageRecognizer;

namespace ConsoleInterface
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Type the full path to the folder, and image recognition will begin:");
            string path = Console.ReadLine();
            MnistRecognizer.TraverseDirectory(path);
        }
    }
}
