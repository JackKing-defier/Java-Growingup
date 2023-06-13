import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class LeetCode {
    /**
     * 345. 反转字符串中的元音字母
     *
     * @param s
     * @return
     */
    public String reverseVowels(String s) {
        if (s == null) return null;
        final Set<Character> keySet = new HashSet<Character>(Arrays.asList('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'));
        int i = 0;
        int j = s.length() - 1;
        char[] result = new char[s.length()];

        while (i <= j) {
            char ci = s.charAt(i);
            char cj = s.charAt(j);
            if (!keySet.contains(ci)) {
                result[i++] = ci;
            } else if (!keySet.contains(cj)) {
                result[j--] = cj;
            } else {
                result[i++] = cj;
                result[j--] = ci;
            }

        }

        return new String(result);
    }

    public static void main(String args[]) {
        LeetCode excuate = new LeetCode();
        String s = "leetcode";

        System.out.println(excuate.reverseVowels(s));
    }
}
