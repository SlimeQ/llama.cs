public static class Util
{
    public static T[] RemoveAt<T>(this T[] source, int index)
    {
        if (index < 0 || index >= source.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        T[] newArray = new T[source.Length - 1];
        Array.Copy(source, 0, newArray, 0, index);
        Array.Copy(source, index + 1, newArray, index, source.Length - index - 1);

        return newArray;
    }
    
    public static T[] Add<T>(this T[] source, T item)
    {
        T[] newArray = new T[source.Length + 1];
        Array.Copy(source, newArray, source.Length);
        newArray[source.Length] = item;

        return newArray;
    }
}