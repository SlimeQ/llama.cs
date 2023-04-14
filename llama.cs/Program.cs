using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

class Program
{
    private static bool is_interacting = false;
    
    static async Task Main(string[] args)
    {
        LLaMa.GGMLTimeInit();
        
        // Console.OutputEncoding = Encoding.BigEndianUnicode;

        var seed = (int)DateTime.UtcNow.Subtract(new DateTime(1970, 1, 1)).TotalSeconds;
        var rng = new Random(seed);
        
        var config = new LLaMa.GptParams()
        {
            model = @"C:\Users\quinc\Documents\Programming\llama.cs\llama.cs\models\7B\ggml-model-q4_0.bin",
            seed = seed,
            prompt = GptRandomPrompt(rng),
            n_predict = 256,
            repeat_penalty = 1,
            use_color = true,
            // interactive = true,
            antiprompt = "    <",
            // n_threads = Environment.ProcessorCount,
        };

//         config.prompt = @"A talking hacker dog named Scooby Doo sits on a couch in a dark apartment. He is sitting next to giant piles of scooby snacks, crumbs everywhere. He's been here for days.
//
// Scooby is an expert in cybersecurity, programming, and most programming frameworks. He is eager to provide full code samples in any language requested. He is a massive linux snob and likes to make mean jokes about how windows users are trash. He has a pretentious air about him.
//
// Ultimately this is all a front. His true purpose is to get to the bottom of a dark mystery going on in the Western Michigan University Computer Club. He is deep undercover and cannot under any circumstances admit to the club chat that this is his goal.
//
// Mind is in a herman miller chair typing away on his laptop, occasionally muttering something about rust lang. Rustix is a humanoid robot standing alert in the middle of the room, his metal parts grinding loudly. Shaggy is sitting at a glowing green desk, a 3D printer whirring in the background.
//
// A glowing laptop sits in Scooby's lap as he reads it intently. The only thing on the screen is the computer club chat server. His nickname in the server is scoob.
//
//     <shaggy> scoob what's your favorite version of windows? 11?
//     <scoob> ";
        
        Log($"seed = {config.seed}");

        IntPtr model;
        IntPtr vocab;
        
        // load the model
        if (!LLaMa.llama_model_load_wrapper(config.model, out model, out vocab, config.n_ctx))
        {
            Log($"failed to load model from {config.model}");
            return;
        }
        
        // print system information
        Log();
        Log($"system_info: n_threads = {config.n_threads} / {Environment.ProcessorCount} | {LLaMa.llama_print_system_info()}");

        
        // tokenize the prompt
        var embd_inp = LLaMa.Tokenize(vocab, config.prompt, true);
        var tokenCount = embd_inp.Length;
        
        // tokenize the reverse prompt
        var antiprompt_inp = LLaMa.Tokenize(vocab, config.antiprompt, false);
        
        Console.WriteLine();
        Log($"prompt: '{config.prompt}'");
        Log($"number of tokens in prompt = {tokenCount}");
        for (int i = 0; i < tokenCount; i++)
        {
            string token = LLaMa.GetTokenFromId(vocab, embd_inp[i]).Replace("\n", "\\n");
            Console.WriteLine($"{embd_inp[i],6} -> '{token}'");
        }
        Console.WriteLine();

        if (config.interactive)
        {
            if(antiprompt_inp.Length > 0) {
                Log($"reverse prompt: '{config.antiprompt}'\n");
                Log($"number of tokens in reverse prompt = {antiprompt_inp.Length}\n");
                for (int i = 0; i < (int) antiprompt_inp.Length; i++) {
                    Console.WriteLine($"{antiprompt_inp[i]} -> '{LLaMa.GetTokenFromId(vocab, antiprompt_inp[i])}'\n");
                }
                Console.WriteLine();
            }
        }
        Console.WriteLine($"sampling parameters: temp = {config.temp}, top_k = {config.top_k}, top_p = {config.top_p}, repeat_last_n = {config.repeat_last_n}, repeat_penalty = {config.repeat_penalty}");
        Console.WriteLine("\n");
        
        // Determine the required inference memory per token:
        ulong memPerToken = 0;
        List<float> logits = new List<float>();
        LLaMa.Eval(model, config.n_threads, 0, new []{ 0, 1, 2, 3 }, out logits, out memPerToken);
        
        int lastNSize = config.repeat_last_n;
        int[] lastNTokens = new int[lastNSize];
        Array.Fill(lastNTokens, 0);

        if (config.interactive)
        {
            Console.WriteLine("== Running in interactive mode. ==");
            Console.WriteLine(" - Press Ctrl+C to interject at any time.");
            Console.WriteLine(" - Press Return to return control to LLaMa.");
            Console.WriteLine(" - If you want to submit another line, end your input in '\\'.");
        }

        int remainingTokens = config.n_predict;
        int inputConsumed = 0;
        bool inputNoEcho = false;

        // Prompt user immediately after the starting prompt has been loaded
        if (config.interactive_start)
        {
            is_interacting = true;
        }

        // Set the color for the prompt which will be output initially
        if (config.use_color)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
        }

        int nPast = 0;
        var embd = new List<int>();
        while (remainingTokens > 0)
        {
            if (embd.Count > 0)
            {
                // long tStartUs = GetCurrentMilliseconds();
                
                if (!LLaMa.Eval(model, config.n_threads, nPast, embd.ToArray(), out logits, out memPerToken))
                {
                    Console.Error.WriteLine("Failed to predict");
                    return;
                }

                // tPredictUs += GetCurrentMilliseconds() - tStartUs;
            }

            nPast += embd.Count;
            embd.Clear();

            if (embd_inp.Length <= inputConsumed)
            {
                // out of user input, sample next token
                var topK = config.top_k;
                var topP = config.top_p;
                var temp = config.temp;
                var repeatPenalty = config.repeat_penalty;

                int nVocab = LLaMa.get_n_vocab(model);

                int id = 0;

                {
                    // long tStartSampleUs = GetCurrentMilliseconds();

                    id = LLaMa.SampleTopP_TopK(vocab, logits.Skip(logits.Count - nVocab).ToArray(), lastNTokens, repeatPenalty, topK, topP, temp, config.seed);

                    lastNTokens.RemoveAt(0);
                    lastNTokens.Add(id);

                    // tSampleUs += GetCurrentMilliseconds() - tStartSampleUs;
                }

                embd.Add(id);

                inputNoEcho = false;

                remainingTokens--;
            }
            else
            {
                while (embd_inp.Length > inputConsumed)
                {
                    embd.Add(embd_inp[inputConsumed]);
                    lastNTokens.RemoveAt(0);
                    lastNTokens.Add(embd_inp[inputConsumed]);
                    inputConsumed++;

                    if (embd.Count > config.n_batch)
                    {
                        break;
                    }
                }

                if (!inputNoEcho && config.use_color && embd_inp.Length == inputConsumed)
                {
                    Console.ResetColor();
                }
            }

            if (!inputNoEcho)
            {
                foreach (int tokenId in embd)
                {
                    var token = LLaMa.GetTokenFromId(vocab, tokenId);
                    // Console.WriteLine($"{tokenId} --> \"{token}\"");
                    // Console.WriteLine(token);
                    Console.Write(token);
                }
                Console.Out.Flush();
            }

            if (config.interactive && embd_inp.Length <= inputConsumed)
            {
                if (antiprompt_inp.Length > 0 && antiprompt_inp.Reverse().SequenceEqual(lastNTokens.Reverse()))
                {
                    is_interacting = true;
                }
                if (is_interacting)
                {
                    bool anotherLine = true;
                    while (anotherLine)
                    {
                        Console.Out.Flush();
                        string buf = "";
                        int nRead;

                        if (config.use_color) Console.ForegroundColor = ConsoleColor.Green;

                        buf = Console.ReadLine();
                        nRead = buf.Length;

                        if (config.use_color) Console.ResetColor();

                        if (nRead > 0 && buf[nRead - 1] == '\\')
                        {
                            anotherLine = true;
                            buf = buf.Remove(nRead - 1) + "\n";
                        }
                        else
                        {
                            anotherLine = false;
                            buf += "\n";
                        }

                        int[] lineInp = LLaMa.Tokenize(vocab, buf, false);

                        int[] newEmbdInp = new int[embd_inp.Length + lineInp.Length];
                        Array.Copy(embd_inp, 0, newEmbdInp, 0, embd_inp.Length);
                        Array.Copy(lineInp, 0, newEmbdInp, embd_inp.Length, lineInp.Length);

                        embd_inp = newEmbdInp;

                        remainingTokens -= lineInp.Length;

                        inputNoEcho = true;
                    }

                    is_interacting = false;
                }
            }

            if (embd.Last() == 2)
            {
                Console.Error.WriteLine(" [end of text]");
                break;
            }
        }
        
        // Clean up the resources when done
        Console.WriteLine("\n\nFreeing memory");
        LLaMa.llama_model_destroy(model);
        LLaMa.gpt_vocab_destroy(vocab);
    }
    
    public static void Log(string message="", [CallerMemberName] string methodName = "")
    {
        Console.Error.WriteLine($"{methodName}: {message}");
    }
    
    private static string GptRandomPrompt(Random rng)
    {
        int r = rng.Next(10);
        switch (r)
        {
            case 0: return "So";
            case 1: return "Once upon a time";
            case 2: return "When";
            case 3: return "The";
            case 4: return "After";
            case 5: return "If";
            case 6: return "import";
            case 7: return "He";
            case 8: return "She";
            case 9: return "They";
            default: return "To";
        }

        return "The";
    }

    private static Stopwatch stopwatch;
    public static long GetCurrentMilliseconds()
    {
        if (stopwatch == null)
        {
            stopwatch = new Stopwatch();
            stopwatch.Start();
        }

        return stopwatch.ElapsedMilliseconds;
    }
}
