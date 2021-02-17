using ImageRecognizer;
using System.Threading.Tasks;

namespace GraphicalInterface
{
    class ViewModel
    {
        public string[] ImageLabels { get; private set;}

        public Task RecognitionTask { get; set; }

        public ViewModel()
        {
            ImageLabels = MnistRecognizer.ImageLabels;
            RecognitionTask = null;
        }
    }
}
