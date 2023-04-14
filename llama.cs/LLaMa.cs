using System.Runtime.InteropServices;
using System.Text;

public static class LLaMa
{
    public class GptParams
    {
        public int seed = -1; // RNG seed
        public int n_threads = Math.Min(4, Environment.ProcessorCount);
        public int n_predict = 128; // new tokens to predict
        public int repeat_last_n = 64;  // last n tokens to penalize
        public int n_ctx = 512; //context size

        // Sampling parameters
        public int top_k = 40;
        public float top_p = 0.95f;
        public float temp = 0.8f;
        public float repeat_penalty = 1.3f;

        public int n_batch = 8; // batch size for prompt processing

        public string model = "models/lamma-7B/ggml-model.bin"; // model path
        public string prompt;

        public bool use_color = false; // use color to distinguish generations and inputs

        public bool interactive = false; // interactive mode
        public bool interactive_start = false; // reverse prompt immediately
        public string antiprompt = ""; // string upon seeing which more user input is prompted
    }
    
    [DllImport("libs/llamalib.dll")]
    private static extern int ggml_time_init_wrapper();
    public static void GGMLTimeInit()
    {
        ggml_time_init_wrapper();
    }
    
    [DllImport("libs/llamalib.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern bool llama_model_load_wrapper(string fname, out IntPtr model, out IntPtr vocab, int n_ctx);

    [DllImport("libs/llamalib.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void llama_model_destroy(IntPtr model);

    [DllImport("libs/llamalib.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void gpt_vocab_destroy(IntPtr vocab);
    
    [DllImport("libs/llamalib.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern string llama_print_system_info();

    public static int[] Tokenize(IntPtr vocab, string text, bool bos)
    {
        int[] output = new int[32000];
        int output_length;
        llama_tokenize_wrapper(vocab, text, bos, output, out output_length);
        
        int[] tokenized = new int[output_length];
        Array.Copy(output, tokenized, output_length);
        
        return tokenized;
    }
    
    [DllImport("libs/llamalib.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern void llama_tokenize_wrapper(IntPtr vocab, string text, bool bos, [Out] int[] output, out int output_length);
    
    [DllImport("libs/llamalib.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr llama_id_to_token(IntPtr vocab, int i);

    public static string GetTokenFromId(IntPtr vocab, int i)
    {
        IntPtr tokenPtr = llama_id_to_token(vocab, i);
        return Marshal.PtrToStringUTF8(tokenPtr);
    }
    

    public static bool Eval(
        IntPtr model,
        int n_threads,
        int n_past,
        int[] embd_inp,
        out List<float> embd_w,
        out ulong mem_per_token)
    {
        int embd_inp_size = embd_inp.Length;
        int vocab_size = get_n_vocab(model);
        int embd_w_size = embd_inp_size * vocab_size;
        float[] embd_w_array = new float[embd_w_size];

        bool result = llama_eval_wrapper(model, n_threads, n_past, embd_inp, embd_inp_size, embd_w_array, embd_w_size, out mem_per_token);

        embd_w = new List<float>(embd_w_array);

        return result;
    }
    
    [DllImport("libs/llamalib.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern bool llama_eval_wrapper(
        IntPtr model,
        int n_threads,
        int n_past,
        int[] embd_inp,
        int embd_inp_size,
        float[] embd_w,
        int embd_w_size,
        out ulong mem_per_token);
    
    [DllImport("libs/llamalib.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern int get_n_vocab(IntPtr model);
    
    [DllImport("libs/llamalib.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern int llama_sample_top_p_top_k_wrapper(IntPtr vocab, float[] logits, int[] last_n_tokens, int last_n_tokens_size, double repeat_penalty, int top_k, double top_p, double temp, int seed, out int result);
    
    public static int SampleTopP_TopK(IntPtr vocab, float[] logits, int[] last_n_tokens, double repeat_penalty, int top_k, double top_p, double temp, int seed)
    {
        int result;
        int success = llama_sample_top_p_top_k_wrapper(vocab, logits, last_n_tokens, last_n_tokens.Length, repeat_penalty, top_k, top_p, temp, seed, out result);

        if (success == 0)
        {
            throw new Exception("Failed to sample from logits.");
        }

        return result;
    }
    
    public static string TokenIdsToText(IntPtr vocab, List<int> tokenIds)
    {
        StringBuilder sb = new StringBuilder();
        foreach (int tokenId in tokenIds)
        {
            int bufferSize = 256;
            StringBuilder token = new StringBuilder(bufferSize);
            GetTokenFromId(vocab, tokenId);
            sb.Append(token);
        }
        return sb.ToString();
    }
    
    
}