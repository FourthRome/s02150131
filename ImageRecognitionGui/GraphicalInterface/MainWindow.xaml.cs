using System.Threading.Tasks;
using System.Windows;
using System.Windows.Forms;
using System.Diagnostics;
using ImageRecognizer;

namespace GraphicalInterface
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private ViewModel viewModel;
        //---------------------------------------------
        // Constructors + window's basic event handlers
        //---------------------------------------------
        public MainWindow()
        {
            InitializeComponent();
            viewModel = new ViewModel();
            DataContext = viewModel;
        }

        public void OnClickStart(object sender, RoutedEventArgs e)
        {
            Trace.WriteLine("Before starting recognition");
            string par = directoryPathTextBox.Text;
            viewModel.RecognitionTask = Task.Run(async () => {
                await MnistRecognizer.ProcessImagesInDirectory(par, RecognitionCallback); 
            });
            Trace.WriteLine("Recognition must have started");
        }

        public void OnClickStop(object sender, RoutedEventArgs e)
        {
            MnistRecognizer.StopProcessing();
        }

        public void OnClickChooseFolder(object sender, RoutedEventArgs e)
        {
            FolderBrowserDialog dlg = new FolderBrowserDialog();

            if (dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                directoryPathTextBox.Text = dlg.SelectedPath;
            }
        }

        public void RecognitionCallback(RecognitionResult result)
        {
            viewModel.RecognitionResults.Add(result);
            Trace.WriteLine("Hooray!");
        }

    }
}
