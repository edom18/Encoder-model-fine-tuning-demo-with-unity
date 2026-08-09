public interface ITokenizer
{
    (int[] ids, int[] mask) Encode(string text, int maxLength);
}