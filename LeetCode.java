import javax.swing.tree.TreeNode;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;


public class LeetCode {

    public class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    // 树定义
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }


    /**
     * 345. 反转字符串中的元音字母
     */
    public String reverseVowels(String s) {
        if (s == null) {
            return null;
        }
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

    public int findKthLargest(int[] nums, int k) {

        PriorityQueue<Integer> pq = new PriorityQueue<Integer>();
        if (nums.length < 1) {
            return 0;
        }
        for (int e : nums) {
            pq.add(e);
            if (pq.size() > k) {
                pq.poll();
            }
        }
        return pq.peek();
    }

    /**
     * 70. 爬楼梯
     */
    public int climbStairs(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    public int climbStairs2(int n) {
        if (n <= 2) {
            return n;
        }
        int pre2 = 1;
        int pre1 = 2;
        for (int i = 2; i < n; i++) {
            int cur = pre1 + pre2;
            pre2 = pre1;
            pre1 = cur;
        }
        return pre1;//因为此处的 pre1 已经是 cur
    }

    /**
     * 213. 打家劫舍 II
     */
    public int rob(int[] nums) {
        int pre2 = 0, pre1 = 0;

        for (int i = 0; i < nums.length; i++) {
            int cur = Math.max(pre2 + nums[i], pre1);
            pre2 = pre1;
            pre1 = cur;
        }
        return pre1;
    }

    /**
     * rob2
     */
    public int rob2(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        return Math.max(
                rob(Arrays.copyOfRange(nums, 0, nums.length - 1))
                , rob(Arrays.copyOfRange(nums, 1, nums.length)));
    }

    /**
     * 64. 最小路径和
     */
    public int minPathSum(int[][] grid) {
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (i == 0 && j == 0) {
                    continue;
                } else if (i == 0) {
                    grid[0][j] = grid[0][j - 1] + grid[0][j];
                } else if (j == 0) {
                    grid[i][0] = grid[i - 1][0] + grid[i][0];
                } else {
                    grid[i][j] = Math.min(grid[i - 1][j], grid[i][j - 1]) + grid[i][j];
                }
            }
        }
        return grid[grid.length - 1][grid[0].length - 1];
    }

    /**
     * 62. 不同路径
     * 使用杨辉三角技巧，看图可知
     */
    public int uniquePaths(int m, int n) {
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[j] += dp[j - 1];
            }
        }
        return dp[n - 1];
    }

    /**
     * 303. 区域和检索 - 数组不可变
     */
    class NumArray {

        private int[] sum;

        public NumArray(int[] nums) {
            sum = new int[nums.length + 1];
            for (int i = 0; i < nums.length; i++) {
                sum[i + 1] = sum[i] + nums[i];
            }
        }

        public int sumRange(int left, int right) {
            return sum[right + 1] - sum[left];
        }
    }

    //53. 最大子数组和
    public int maxSubArray(int[] nums) {
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        for (int i = 1; i < nums.length; i++) {
            dp[i] = Math.max(nums[i], dp[i - 1] + nums[i]);
        }

        int res = dp[0];
        for (int j = 0; j < nums.length; j++) {
            res = Math.max(res, dp[j]);
        }
        return res;

    }

    //646. 最长数对链
    public int findLongestChain(int[][] pairs) {
        Arrays.sort(pairs, (a, b) -> a[0] - b[0]);
        int[] dp = new int[pairs.length];
        Arrays.fill(dp, 1);
        int ret = dp[0];
        for (int i = 0; i < pairs.length; i++) {
            for (int j = 0; j < i; j++) {
                if (pairs[j][1] < pairs[i][0]) dp[i] = Math.max(dp[i], dp[j] + 1);
            }
            ret = Math.max(ret, dp[i]);
        }
        return ret;
    }

    // 376. 摆动序列
    // 参考讲解：https://leetcode.cn/problems/wiggle-subsequence/solution/bai-dong-xu-lie-by-leetcode-solution-yh2m/
    public int wiggleMaxLength(int[] nums) {
        if (nums.length < 2) {
            return nums.length;
        }
        int up = 1;
        int down = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > nums[i - 1]) {
                up = down + 1;
            }
            if (nums[i] < nums[i - 1]) {
                down = up + 1;
            }
        }
        return Math.max(up, down);
    }

    // 1143. 最长公共子序列
    // String的index是从0开始
    public int longestCommonSubsequence(String text1, String text2) {
        int n = text1.length();
        int m = text2.length();
        int[][] dp = new int[n + 1][m + 1];
        dp[0][0] = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);
                }
            }
        }
        return dp[n][m];
    }

    //剑指 Offer 63. 股票的最大利润
    // 121. 买卖股票的最佳时机
    public int maxProfit(int[] prices) {
        int cost = Integer.MAX_VALUE, profit = 0;
        for (int price : prices) {
            cost = Math.min(cost, price);
            profit = Math.max(profit, price - cost);
        }
        return profit;
    }
//
//    public int maxProfit(int[] prices) {
//        int minBuy = prices[0];
//        int profix = 0;
//        for (int i = 0; i < prices.length; i++) {
//            minBuy = Math.min(prices[i], minBuy);
//            profix = Math.max(profix, prices[i] - minBuy);
//        }
//        return profix;
//    }

    // 122. 买卖股票的最佳时机 II
    public int maxProfit2(int[] prices) {
        int profit = 0;
        for (int i = 0; i < prices.length - 1; i++) {
            if (prices[i + 1] > prices[i]) {
                profit += (prices[i + 1] - prices[i]);
            }
        }
        return profit;
    }

    //583. 两个字符串的删除操作
    public int minDistance(String word1, String word2) {
        int n = word1.length();
        int m = word2.length();
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return m + n - dp[n][m] * 2;
    }

    //太难了还不会做：
    public int minDistance2(String word1, String word2) {
        int n = word1.length();
        int m = word2.length();
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return n - dp[n][m];
    }


    //110. 平衡二叉树
    public boolean isBalanced(TreeNode root) {
        if (root == null) {
            return true;
        }
        // 平衡二叉树除了当前跟节点下的子树高度不超过1，同时子树也需要是平衡二叉树，所以需要递归
        return (Math.abs(maxDepth(root.left) - maxDepth(root.right)) <= 1) && isBalanced(root.left) && isBalanced(root.right);

    }

    private int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    //572. 另一棵树的子树
    public boolean isSubtree(TreeNode root, TreeNode subRoot) {
        if (root == null || subRoot == null) return false;
        return isSubtree(root.left, subRoot) || isSubtree(root.right, subRoot) || isSubtreeStartRoot(root, subRoot);
    }

    private boolean isSubtreeStartRoot(TreeNode root, TreeNode subRoot) {
        if (root == null && subRoot == null) return true;
        if (root == null || subRoot == null) return false;
        if (root.val != subRoot.val) {
            return false;
        }
        return isSubtreeStartRoot(root.left, subRoot.left) && isSubtreeStartRoot(root.right, subRoot.right);
    }

    //101. 对称二叉树
    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return isSymmetric(root.left, root.right);

    }

    private boolean isSymmetric(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null) return true;
        if (root1 == null || root2 == null) return false;
        if (root1.val != root2.val) return false;
        return isSymmetric(root1.left, root2.right) && isSymmetric(root1.right, root2.left);
    }

    //  102. 二叉树的层序遍历
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> allTree = new ArrayList<List<Integer>>();
        Queue<TreeNode> queue = new LinkedList<TreeNode>();

        if (root == null) {
            return allTree;
        }
        queue.offer(root);

        while (!queue.isEmpty()) {
            List<Integer> perLevel = new ArrayList<Integer>();
            int n = queue.size();
            for (int i = 0; i < n; i++) {
                TreeNode node = queue.poll();
                perLevel.add(node.val);
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            allTree.add(perLevel);
        }
        return allTree;
    }

    // 242. 有效的字母异位词
    public boolean isAnagram(String s, String t) {
        int[] cnt = new int[26];
        for (char c : s.toCharArray()) {
            cnt[c - 'a']++;
        }
        for (char c : t.toCharArray()) {
            cnt[c - 'a']--;
        }
        for (int i : cnt) {
            if (i != 0) {
                return false;
            }
        }
        return true;
    }

    // 205. 同构字符串(需要正反映射)
    public boolean isIsomorphic(String s, String t) {
        Map<Character, Character> solutionMap = new HashMap<Character, Character>();
        for (int i = 0; i < s.length(); i++) {
            if (solutionMap.containsKey(s.charAt(i))) {
                if (solutionMap.get(s.charAt(i)) == t.charAt(i)) {
                    continue;
                } else {
                    return false;
                }
            } else {
                solutionMap.put(s.charAt(i), t.charAt(i));
            }
        }
        return true;
    }

    // 674. 最长连续递增序列
    public int findLengthOfLCIS(int[] nums) {
        if (nums.length < 2) {
            return nums.length;
        }

        int maxLength = 1;
        int perLength = 1;
        for (int j = 1; j < nums.length; j++) {
            if (nums[j - 1] < nums[j]) {
                perLength += 1;
            } else {
                perLength = 1;
            }
            maxLength = Math.max(perLength, maxLength);
        }

        return maxLength;
    }

    // 673. 最长递增子序列的个数
    public int findNumberOfLIS(int[] nums) {
        // dp[i]是以nums[i]结尾的递增子序列长度
        // Map<Integer, Integer> dp[i], 个数。
        // 寻找最大的dp[i]的映射Integer
        if (nums.length < 2) {
            return nums.length;
        }

        int LISLength = 1;
        int dp[] = new int[nums.length];
        Map<Integer, Integer> LISMap = new HashMap<Integer, Integer>();
        for (int i = 0; i < nums.length; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = dp[j] + 1;
                }
            }
            LISLength = Math.max(LISLength, dp[i]);
            LISMap.put(dp[i], LISMap.getOrDefault(dp[i], 0) + 1);
        }
        return LISMap.get(LISLength);
    }


    public static void quickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pivotIndex = partion(arr, low, high);
            quickSort(arr, low, pivotIndex - 1);
            quickSort(arr, pivotIndex + 1, high);
        }
    }

    //手写切分函数
    public static int partion(int[] nums, int left, int right) {
        int key = nums[left];
        int i = left;
        int j = right + 1;
        while (true) {
            while (nums[++i] < key) {
                if (i == right) break;
            }
            while (nums[--j] > key) {
                if (left == j) break;
            }
            if (i >= j) break;
            swap(nums, j, i);
        }
        swap(nums, left, j);
        return j;
    }

    // ChatGPT 版本切分函数，不太好理解，一样有效
    public static int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;

        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                swap(arr, i, j);
            }
        }

        swap(arr, i + 1, high);
        return i + 1;
    }

    public static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    public static void printArray(int[] arr) {
        for (int num : arr) {
            System.out.print(num + " ");
        }
        System.out.println();
    }

    //455. 分发饼干
    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int j = 0;
        //针对每个饼干，适配小朋友，从小到大开始
        for (int i = 0; i < s.length & j < g.length; i++) {
            if (s[i] >= g[j]) {
                j++;
            }
        }
        return j;
    }

    //665. 非递减数列
    public boolean checkPossibility(int[] nums) {
        int len = nums.length;
        if (len <= 1) return true;
        int flag = 0;
        for (int i = 1; i < len && flag < 2; i++) {
            if (nums[i - 1] <= nums[i]) {
                continue;
            }
            flag++;
            if (i - 2 >= 0 && nums[i - 2] > nums[i]) {
                nums[i] = nums[i - 1];
            } else {
                nums[i - 1] = nums[i];
            }
        }
        return flag <= 1;
    }

    //374. 猜数字大小
    public int guessNumber(int n, int target) {
        int l = 0;
        int h = n;
        while (l <= h) {
            int m = l + (h - l) / 2;
            if (m < target) {
                l = m + 1;
            } else if (m > target) {
                h = m - 1;
            } else {
                return m;
            }
        }
        return -1;
    }

    //9. 回文数——不使用字符串转换的话，注意边界条件和细节
    public boolean isPalindrome(int x) {
        if (x < 0) {
            return false;
        }
        int div = 1;
        while (x / div >= 10) div *= 10;
        while (x > 0) {
            int left = x / div;
            int right = x % 10;
            if (left != right) return false;
            x = (x % div) / 10;
            div /= 100;
        }
        return true;
    }

    // 11. 盛最多水的容器
    public int maxArea(int[] height) {
        int n = height.length;
        int ans = 0;
        int i = 0, j = n - 1;
        while (i < j) {
            if (height[i] >= height[j]) {
                ans = Math.max(ans, (j - i) * Math.min(height[i], height[j]));
                j--;
            } else {
                ans = Math.max(ans, (j - i) * Math.min(height[i], height[j]));
                i++;
            }
        }
        return ans;
    }

    //
    /*
    Input:
s = "abcxyz123"
dict = ["abc","123"]
Output:
"<b>abc</b>xyz<b>123</b>"
Input:
s = "aaabbcc"
dict = ["aaa","aab","bc"]
Output:
"<b>aaabbc</b>c"，
dict内字符串在s中只有一次，infexOf
    * */

    public String boldString(String s, ArrayList<String> dict) {
        StringBuilder ans = new StringBuilder();
        int n = s.length();
        int[] flag = new int[n];
        int start = 0;
        int end = 0;
//        List<List<Integer>> pair = new ArrayList<>();
        //Pair
        for (int i = 0; i < dict.size(); i++) {
            start = s.indexOf(dict.get(i));
            if (start < 0) {
                continue;
            }
            end = start + dict.get(i).length();

            for (int j = start; j < end; j++) {
                flag[j] = 1;
            }
//            List <Integer> temp = new ArrayList<>();
//            temp.add(start);
//            temp.add(end);
        }

        String boldStart = "<b>";
        String boldEnd = "</b>";
        Boolean f = false;
        for (int i = 0; i < n; i++) {
            if (flag[i] == 1) {
                if (i > 0 && flag[i - 1] == 0) {
                    ans.append(boldStart);
                    ans.append(s.charAt(i));
                } else if (i == 0) {
                    ans.append(boldStart);
                    ans.append(s.charAt(i));
                } else if (flag[i - 1] == 1) {
                    ans.append(s.charAt(i));
                }
            } else {
                // flag[i] == 0
                if (i > 0 && flag[i - 1] == 1) {
                    ans.append(boldEnd);
                    ans.append(s.charAt(i));
                } else if (i == 0) {
                    ans.append(s.charAt(i));
                } else if (flag[i - 1] == 0) {
                    ans.append(s.charAt(i));
                }

            }
        }
        return ans.toString();
    }

    //3. 无重复字符的最长子串
    public int lengthOfLongestSubstring(String s) {
        //使用set判定是否存在对应字符，然后刷新int ans
        Set<Character> tempSet = new HashSet<>();
        int ans = 1;
        int res = 1;
        for (int i = 0; i < s.length(); i++) {
            if (tempSet.contains(s.charAt(i))) {
                ans = 1;
                tempSet.removeAll(tempSet);

            }
            tempSet.add(s.charAt(i));
            res = Math.max(res, ans);
            ans++;

        }
        return res;
    }
    /*
    * def json_diff(dict1, dict2):
// """
// Input:
//   two dictionaries.
//   The dictionary value(s) can be string, integer, or dictionary.
//   The dictionary keys is always a string.

// ---

// Output:
//   list of results containing a tuple of (key, dict1[key], dict2[key]) if dict1[key] != dict2[key] for a given key.

// ---

// Example 1:
// 	dict1 = {"a": 1, "b": 3}
// 	dict2 = {"a": 2}
// 	output = [ ("a", 1, 2), ("b", 3, ) ]

// Example 2:
// 	dict1 = {"a": 1}
// 	dict2 = {"a": 1}
// 	output = []

// ---

// Pesudo-code is acceptable.

// You will be assessed on how you approach the problem, not on how accurate your syntax is.

// For example, the following is acceptable:
//   if object is dict then
//     for key and value in dict object
//       if value is not list or dict then
//         add to output (key, value, ...)
//       else
//         ...
//   else if object is list then
//     ...
//   else
//     ...
//   return output
// """


*
* // 	dict1 = {"a": {"e": 5, "d": 2}, "b": 3}
// 	dict2 = {"a": {"e": 2, "d": 2}}
* // 	output = [ ("a.e",5, 2), ("b", 3, ) ]    *
    * */
//    public Map<String, List<Object>> compareDict(Map<String, Object> dict1, Map<String, Object> dict2) {
//        //Map
//        Map<String, List<Object>> result = new HashMap<>();
//
//        for(Map.Entry<String,Object> d1 : dict1.entrySet()) {
//
//            if(!dict2.containsKey(d1.getKey())) {
//                List<Object> tempList = new ArrayList<>();
//                tempList.add((object)d1.getKey(), null);
//                result.put(d1.getKey(), tempList);
//            }
//
//            if(!dict2.get(d1.getKey()).equals(d1.getValue())) {
//                // todo dict
//
//                List<Object> tempList = new ArrayList<>();
//                tempList.add((object)d1.getKey(), dict2.get(d1.getKey()));
//                result.put(d1.getKey(), tempList);
//            }
//        }
//
//        for(Map.Entry<String,Object> d2 : dict2.entrySet()) {
//
//            if (!dict2.containsKey(d2.getKey())) {
//                List<Object> tempList = new ArrayList<>();
//                tempList.add(null, (object) d2.getKey());
//                result.put(d2.getKey(), tempList);
//            }
//        }
//        //key
//
//
//    }

    //
    public ListNode mergeKLists(ListNode[] lists) {
        ListNode dummyNode = new ListNode(-1);
        ListNode p = dummyNode;
        PriorityQueue<ListNode> pq = new PriorityQueue<>(lists.length, (a, b) -> (a.val - b.val));

        for (ListNode list : lists) {
            if (list != null) {
                pq.add(list);
            }
        }

        while (!pq.isEmpty()) {
            ListNode node = pq.poll();
            p.next = node;
            p = p.next;
            if (node.next != null) {
                pq.add(node.next);
            }
        }
        return dummyNode.next;

    }

    //56. 合并区间
    public int[][] merge(int[][] intervals) {
        //思路：一遍扫描将区间置为1，第二遍输出
        int n = intervals.length;
        //int m = intervals[0].length;
        int[][] result = new int[n][2];
        int[] temp = new int[intervals[n - 1][1] + 1];

        for (int i = 0; i < n; i++) {
            int begain = intervals[i][0];
            while (begain <= intervals[i][1]) {
                temp[begain] = 1;
                begain++;
            }
        }
        //如何录入输出，需要注意技巧
        for (int j = 0; j < n + 1; j++) {
//            if () {
//
//            }
        }
        return null;

    }

    //560. 和为 K 的子数组
    public int subarraySum(int[] nums, int k) {
        // 初步思路双指针滑动窗口
        int left = 0, right = 0;
        int n = nums.length;
        int ans = 0;
        while (left < n && right < n) {
            if (left <= right) {
                right++;
            }
            int temp = 0;
            for (int i = left; i <= right; i++) {
                temp += nums[i];
            }
            if (temp == k) {
                ans++;
            } else if (temp < k) {
                right++;
            } else {
                left++;
            }

        }
        return ans;
    }

    //1456. 定长子串中元音的最大数目
    public int maxVowels(String s, int k) {
        int ans = 0;
        for (int i = 0; i < s.length() - k + 1; i++) {
            int right = i + k;
            String str = s.substring(i, right);
            int tempCal = charNumber(str);
            ans = Math.max(ans, tempCal);
        }
        return ans;
    }

    public int charNumber(String t) {
        Set tagSet = new HashSet<>(Arrays.asList('a', 'e', 'i', 'o', 'u'));
        int res = 0;
        for (int i = 0; i < t.length(); i++) {
            if (tagSet.contains(t.charAt(i))) {
                res++;
            }
        }
        return res;
    }

    //238. 除自身以外数组的乘积
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] L = new int[n];
        L[0] = 1;
        for (int i = 1; i < n; i++) {
            L[i] = L[i - 1] * nums[i - 1];
        }

        int[] R = new int[n];
        R[n - 1] = 1;
        for (int j = n - 2; j >= 0; j--) {
            R[j] = nums[j + 1] * R[j + 1];
        }

        int[] ans = new int[n];
        for (int k = 0; k < n; k++) {
            ans[k] = L[k] * R[k];
        }
        return ans;
    }

    //443. 压缩字符串——FUCK！
    public int compress(char[] chars) {
        int n = chars.length;
        if (n < 2) {
            return n;
        }
        int checkIndex = 0;
        int write = 0;
        for (int i = 1; i < n; i++) {
            int count = i - checkIndex;
            if (i == n - 1 || chars[i] != chars[i - 1]) {
                chars[write] = chars[i - 1];
                if (count > 1) {
                    char[] countArray = String.valueOf(count).toCharArray();
                    for (int j = 0; j < countArray.length; j++) {
                        chars[write + j] = countArray[j];
                    }
                    write += countArray.length;
                }
                checkIndex = i;
                //chars[write] = chars[i]; //
            }
        }
        System.out.println(chars);
        return ++write;
    }

    //2215. 找出两数组的不同
    public List<List<Integer>> findDifference(int[] nums1, int[] nums2) {
        Set<Integer> set1 = new HashSet<>();
        Set<Integer> set2 = new HashSet<>();
        for (int num : nums1) {
            set1.add(num);
        }
        for (int num : nums2) {
            set2.add(num);
        }
        Set<Integer> tempSet = new HashSet<>(set1);
        set1.removeAll(set2);
        set2.removeAll(tempSet);

        List<List<Integer>> ans = new ArrayList<>();
        ans.add(new ArrayList<>(set1));
        ans.add(new ArrayList<>(set2));
        return ans;
    }

    //541. 反转字符串 II
    public String reverseStr(String s, int k) {
        int start = -1, left = start + 1, right = start + k;
        int n = s.length();
        char[] array = s.toCharArray();
        while (right < n) {
            while (left < right) {
                char temp = array[left];
                array[left++] = array[right];
                array[right--] = temp;
            }
            start += 2 * k;
            left = start + 1;
            right = start + k;
        }
        return array.toString();
    }

    //459. 重复的子字符串
    public boolean repeatedSubstringPattern(String s) {
        StringBuilder sb = new StringBuilder();
        int index = 0;
        int count = 0;
        int n = s.length();
        if (n < 2) {
            return false;
        }
        sb.append(s.charAt(0));
        for (int i = 0; i < n; i++) {
            if (index < sb.toString().length() && sb.charAt(index) == s.charAt(i)) {
                index++;
                if (index == sb.toString().length()) {
                    count++;
                    index = 0;
                }
            } else {
                sb.append(s.charAt(i));
                index = 0;
            }
        }

        int m = sb.toString().length();

        return m != n && n / m == count && n % m == 0;
    }

    //150. 逆波兰表达式求值
    public int evalRPN(String[] tokens) {
        Stack<String> tempStack = new Stack<>();
        for (int i = 0; i < tokens.length; i++) {
            if (isCal(tokens[i])) {
                Integer first = Integer.valueOf(tempStack.pop());
                Integer second = Integer.valueOf(tempStack.pop());
                switch (tokens[i]) {
                    case "+" -> tempStack.push(String.valueOf(first + second));
                    case "-" -> tempStack.push(String.valueOf(first - second));
                    case "*" -> tempStack.push(String.valueOf(first * second));
                    default -> tempStack.push(String.valueOf(first / second));
                }
            } else {
                tempStack.push(tokens[i]);
            }
        }
        return Integer.valueOf(tempStack.pop());

    }

    private boolean isCal(String str) {
        return str.equals("+") || str.equals("-") || str.equals("*") || str.equals("/");
    }

    //872. 叶子相似的树
    public boolean leafSimilar() {
        TreeNode root1 = new TreeNode(1);
        root1.left = new TreeNode(2);
        root1.right = new TreeNode(200);
        TreeNode root2 = new TreeNode(1);
        root2.left = new TreeNode(2);
        root2.right = new TreeNode(200);
        List<Integer> res1 = new ArrayList<>();
        List<Integer> res2 = new ArrayList<>();
        leafList(root1, res1);
        leafList(root2, res2);

        if (res1.size() != res2.size()) return false;
        for (int i = 0; i < res1.size(); i++) {
            System.out.println(res1.get(i));
            System.out.println(res2.get(i));
            if (!res1.get(i).equals(res2.get(i))) return false;
        }
        return true;
    }

    private void leafList(TreeNode root, List<Integer> res) {
        if (root == null) return;
        //System.out.println(root.val);
        if (root.left == null && root.right == null) res.add(root.val);
        leafList(root.left, res);
        leafList(root.right, res);
    }

    //860. 柠檬水找零
    public boolean lemonadeChange(int[] bills) {
        int[] cash = new int[2];
        for (int i = 0; i < bills.length; i++) {
            if (bills[i] == 5) {
                cash[0]++;
            } else if (bills[i] == 10 && cash[0] > 0) {
                cash[0]--;
                cash[1]++;
            } else if (bills[i] == 10 && cash[0] <= 0) {
                return false;
            } else if (bills[i] == 20 && cash[1] > 0 && cash[0] > 0) {
                cash[0]--;
                cash[1]--;
            } else if (bills[i] == 20 && cash[1] <= 0 && cash[0] > 2) {
                cash[0] -= 3;
            } else {
                return false;
            }
        }
        return true;
    }

    //打家劫舍2
    public int newRob2(int[] nums) {
        if (nums.length < 2) return nums[0];
        int result1 = robRange(nums, 0, nums.length - 2);
        int result2 = robRange(nums, 1, nums.length - 1);
        return Math.max(result1, result2);
    }

    private int robRange(int[] nums, int start, int end) {
        int[] dp = new int[end + 1];
        dp[start] = nums[start];
        if (end - start < 1) {
            return nums[start];
        }
        dp[start + 1] = Math.max(nums[start], nums[start + 1]);
        for (int i = start + 2; i <= end; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        return dp[end];
    }

    /*
Write a java multi threading program where:
1. Thread 1 prints "O" for 10 times;
2. Thread 2 prints "K" for 10 times;
3. Thread 3 prints "X" for 10 times;

Then the expected output from the program is:

OKX
OKX
OKX
OKX
OKX
OKX
OKX
OKX
OKX
OKX
*/

//    private void TestOKX (Thread t1) {
//        Lock
//        try {
//            System.out.println("O");
//
//        } finally {
//
//        }
//
//    }
//    private final Lock lock = new ReentrantLock();
//    public void OKXPrint(String str) {
//        lock.lock();
//        try {
//            System.out.print(str);
//        } finally {
//            lock.unlock();
//        }
//    }
    /*
    public static void main(String[] args) throws InterruptedException {
    Object o = new Object();
    Object k = new Object();
    Object x = new Object();

    int count = 10;
    Thread thread1 = new Thread(() -> {
        try {
            int i = 0;
            while (i < count) {
                synchronized (o) {
                    if (i > 0) {
                        o.wait();
                    }
                    System.out.println("o");
                    i++;
                    synchronized (k) {
                        k.notify();
                    }
                }
            }

        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    });
    Thread thread2 = new Thread(() -> {
        try {
            int i = 0;
            while (i < count) {
                synchronized (k) {
                    k.wait();
                    System.out.println("k");
                    i++;
                    synchronized (x) {
                        x.notify();
                    }
                }
            }

        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    });
    Thread thread3 = new Thread(() -> {
        try {
            int i = 0;
            while (i < count) {
                synchronized (x) {
                    x.wait();
                    System.out.println("x");
                    System.out.println();
                    i++;
                    synchronized (o) {
                        o.notify();
                    }
                }
            }

        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    });
    thread1.start();
    thread2.start();
    thread3.start();
    thread3.join();
}
    * */
//    private int n;
//    public Create getInstance() {
//        synchronized () {
//
//        }
//    }

    private Map<Integer, ArrayList<Integer>> nodeMap = new HashMap<Integer, ArrayList<Integer>>();

    public int mixAncester(TreeNode root, int a, int b) {
        int ans = root.val;
        ArrayList<Integer> aAncester = nodeMap.get(a);
        ArrayList<Integer> bAncester = nodeMap.get(b);
        //6,2,4
        //6,2,0
        //find first common
        int point1 = 0;
        int len = Math.min(aAncester.size(), bAncester.size());
        for (int i = 0; i < len; i++) {
            if (aAncester.get(i) != bAncester.get(i)) {
                break;
            } else {
                ans = aAncester.get(i);
            }
        }
        return ans;
    }
//    public void travelTree (TreeNode root, int a, int b) {
//        if (root == null) return;
//        if (root.val == a || root.val == b) {
//
//        }
//        ArrayList<Integer> tempList = nodeMap.getOrDefault(root.val, new ArrayList<Integer>());
//        tempList.add(root.val);
//        nodeMap.put(root.val, tempList);
//        travelTree(root.left);
//        travelTree(root.right);
//    }

    /*
    1.
费用统计,得出每个员工出差总费用及次数：
cat employee.txt
name expense count
zhangsan 8000 1
zhangsan 5000 1
lisi 1000 1
lisi 2000 1
wangwu 1500 1
zhaoliu 6000 1
zhaoliu 2000 1
zhaoliu 3000 1

2.
给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和以及对应的子数组。
子数组 是数组中的一个连续部分。
示例 1：
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
示例 2：
输入：nums = [1]
输出：1
示例 3：
输入：nums = [5,4,-1,7,8]
输出：23
提示：
1 <= nums.length <= 105
-104 <= nums[i] <= 104
    * */

    //res < 0, nums[i]
    private int maxArray(int[] nums) {
        int len = nums.length;
        int sum = 0;
        int ans = 0;
        for (int i = 0; i < len; i++) {
            if (sum < 0) {
                sum = 0;
            }
            sum += nums[i];
            ans = Math.max(ans, sum);
        }
        return ans;
    }

    //
    private Map<String, List<Integer>> calExpense(List<String> file) {
        Map<String, List<Integer>> res = new HashMap<String, List<Integer>>();
        for (int i = 0; i < file.size(); i++) {
            String line = file.get(i);
            String[] lineArray = line.split(" ");
            List<Integer> countList = res.getOrDefault(lineArray[0], new ArrayList<Integer>());
            Integer expense = countList.get(0);
            expense += Integer.valueOf(lineArray[1]);
            countList.set(0, expense);
            Integer times = countList.get(1);
            times += Integer.valueOf(lineArray[2]);
            countList.set(1, times);
            res.put(lineArray[0], countList);
        }
        return res;
    }

    /*
    Write a script that prints the first 10 numbers of the Fibonacci sequence.
The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, usually starting with 0 and 1.
Expected Output: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, …
For example: 0+1=1; 1+1=2; 1+2=3, …
Requirements:
Use any programming language of your choice.
Disable Github Copilot if you have it.
You can initialize with [0,1].
    * */
    //
    private int[] testArc(int num) {
        int[] dp = new int[num + 1];
        dp[0] = 0;
        dp[1] = 1;

        for (int i = 2; i < num; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }

        for (int j = 0; j < num; j++) {
            System.out.println(dp[j]);
        }
        return dp;
    }

    private Set<Integer> hashSet = new HashSet<Integer>();

    public boolean canVisitAllRooms(List<List<Integer>> rooms) {
        int len = rooms.size();

        dfs(rooms, 0);
        hashSet.add(0);
        return hashSet.size() == rooms.size();
    }

    private void dfs(List<List<Integer>> rooms, Integer node) {
        if (hashSet.contains(node)) {
            return;
        }

        for (int i = 0; i < rooms.get(node).size(); i++) {
            Integer nextNode = rooms.get(node).get(i);
            hashSet.add(nextNode);
            dfs(rooms, nextNode);
            // hashSet.remove(nextNode);
        }

    }

    /*
    第一题：一个excel表格 表格头信息 A,B,C.....Z,AA,AB,AC 等，请编写一个程序，输入 一个表头信息，输出列数。
例如:
  输入 AB，输出28这个数字。
  calCol(String a)

第二题：
给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。 k 是一个正整数，它的值小于或等于链表的长度。 如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
示例：
给你这个链表：1->2->3->4->5 当 k = 2 时，应当返回: 2->1->4->3->5 当 k = 3 时，应当返回: 3->2->1->4->5
说明：
* 你的算法只能使用常数的额外空间。
* 你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
    17:14
    * */
    public int calCol(String str) {
        char[] charArray = str.toCharArray();
        int len = charArray.length;
        int ans = 0;
        for (int i = 0; i < len; i++) {
            int num = charArray[i] - 'A' + 1;
            ans = ans * 26 + num;
        }
        return ans;
    }

    public void retLink(ListNode root, int k) {
        ListNode slow = root;
        ListNode fast = slow;
        while (fast.next != null) {
            //cut k
            int count = k;
            while (count > 0 && fast.next != null) {
                fast = fast.next;
                count--;
            }
            ListNode cutHead = slow;
            ListNode newHead = fast.next;
            fast.next = null;

            ListNode newEnd = revertListNode(cutHead);
            newEnd.next = newHead;
            slow = newHead;
            fast = slow;
        }

    }

    public ListNode revertListNode(ListNode root) {
        if (root == null) {
            return null;
        }
        ListNode pre = null;
        ListNode cur = root;
        while (cur.next != null) {
            ListNode tmp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = tmp;
        }
        return root;
    }

    /*
There are n files that need to be installed, named as file 0,1,...,(n-1). For each file there could be dependencies on other files, e.g., (1,0) means that to install file 1, you need to install file 0.
Design a program to output a viable installation order.

input1：int
input2: List<List<Integer>>
output: List<Integer>

example1：2,[[1,0]] >> [0,1]
example2: 4，[[1,0],[2,0],[3,1],[3,2]] >> [0, 1, 2, 3] or [0, 2, 1, 3]
*/
    private Set<Integer> ans;

    public List<Integer> installFile(int n, List<List<Integer>> fileRely) {

        Map<Integer, List<Integer>> rely = new HashMap<Integer, List<Integer>>();
        for (int j = 0; j < fileRely.size(); j++) {
            List<Integer> tempList = rely.getOrDefault(fileRely.get(j).get(1), new ArrayList<Integer>());
            tempList.add(fileRely.get(j).get(0));
            rely.put(fileRely.get(j).get(1), tempList);
        }

        for (int i = 0; i < n; i++) {
            ans = new HashSet<>();
            bfsFile(i, rely);
            if (ans.size() == n) {
                return ans.stream().toList();
            }
        }
        return null;
    }

    private void bfsFile(Integer index, Map<Integer, List<Integer>> relyMap) {
        Queue<Integer> queue = new LinkedList<Integer>();

        queue.add(index);
        while (!queue.isEmpty()) {
            int file = queue.poll();
            ans.add(file);
            if (!ans.contains(file)) {
                continue;
            }
            for (int next : relyMap.get(file)) {
                if (!ans.contains(next)) {
                    queue.add(next);
                    ans.add(next);
                }
            }
        }
    }

    //有序重复数组求第一个不小于target的序号
    //{1,2,3,3,4,5,6}
    //4
    //4
    //3, 2
    public int tt(int[] nums, int target) {
        int right = nums.length;
        int left = 0;

        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] >= target) {
                right = mid;
            }
        }
        return left;
    }
    //text
    // word
    //26

    //973. 最接近原点的 K 个点
    public int[][] kClosest(int[][] points, int k) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((p1, p2) -> p2[0] * p2[0] + p2[1] * p2[1] - p1[0] * p1[0] - p1[1] * p1[1]);

//        for (int[] point : points) {
//            if (pq.size() < k) { // 如果堆中不足 K 个，直接将当前 point 加入即可
//                pq.offer(point);
//            } else if (pq.comparator().compare(point, pq.peek()) > 0) { // 否则，判断当前点的距离是否小于堆中的最大距离，若是，则将堆中最大距离poll出，将当前点加入堆中。
//                pq.poll();
//                pq.offer(point);
//            }
//        }

        for (int[] point : points) {
            pq.offer(point);
            if (pq.size() > k) { // 否则，判断当前点的距离是否小于堆中的最大距离，若是，则将堆中最大距离poll出，将当前点加入堆中。
                pq.poll();
            }
        }

        int[][] ans = new int[pq.size()][2];

        for (int j = 0; j < ans.length; j++) {
            ans[j][0] = pq.peek()[0];
            ans[j][1] = pq.peek()[1];
            pq.remove();
        }
        return ans;
    }

    static class Trie {
        //构建TrieNode对象，记录所有26个字符是否存在以及是否为结尾，理念有点类似布隆过滤器

        class TrieNode {
            boolean end;
            TrieNode[] tns = new TrieNode[26];
        }

        TrieNode root;

        public Trie() {
            root = new TrieNode();
        }

        public void insert(String word) {
            TrieNode p = root;
            for (int i = 0; i < word.length(); i++) {
                int u = word.charAt(i) - 'a';
                if (p.tns[u] == null) p.tns[u] = new TrieNode();
                p = p.tns[u];
            }
            p.end = true;
        }

        public boolean search(String word) {
            TrieNode p = root;
            for (int i = 0; i < word.length(); i++) {
                int u = word.charAt(i) - 'a';
                if (p.tns[u] == null) return false;
                p = p.tns[u];
            }
            return p.end;
        }

        public boolean startsWith(String prefix) {
            TrieNode p = root;
            for (int i = 0; i < prefix.length(); i++) {
                int u = prefix.charAt(i) - 'a';
                if (p.tns[u] == null) return false;
                p = p.tns[u];
            }
            return true;
        }
    }

    //1438. 绝对差不超过限制的最长连续子数组
    public int longestSubarray(int[] nums, int limit) {
        int len = nums.length;
        int dis = 0;
        int[] dp = new int[len];// 以i为结尾的nums子数组最长结果
        for (int i = 0; i < len; i++) {
            int max = Integer.MIN_VALUE;
            int min = Integer.MAX_VALUE;
            for (int j = 0; j <= i; j++) {
                max = Math.max(max, nums[j]);
                min = Math.min(min, nums[j]);
                if (max - min <= limit) {
                    dis = Math.max(dis, j + 1);
                }
            }
            dp[i] = dis;
        }
        return dp[len - 1];
    }

    //10004. Design a Rate Limiting System
    static class RateLimiter {
        private Integer requests;

        private Integer time;

        private Stack<Integer> invokeStack;
        private Stack<Integer> tempStack;

        public RateLimiter(int n, int t) {
            this.requests = n;
            this.time = t;
            this.invokeStack = new Stack<Integer>();
            this.tempStack = new Stack<Integer>();
        }

        public boolean shouldAllow(int timestamp) {
            if (invokeStack.size() < requests - 1) {
                invokeStack.push(timestamp);
                return true;
            }

            int count = 0;
            while (!invokeStack.isEmpty() && timestamp - invokeStack.peek() <= time) {
                tempStack.push(invokeStack.pop());
                count++;
            }

            while (!tempStack.isEmpty()) {
                invokeStack.push(tempStack.pop());
            }

            if (count > requests - 1) {
                return false;
            } else {
                invokeStack.push(timestamp);
                return true;
            }
        }
    }

    public static void main(String[] args) {
        LeetCode exculpate = new LeetCode();

        String text1 = "AB", text2 = "execution";
        int[] nums = {38,42,48,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50};
        int[] nums2 = {2, 4, 6};
        int[][] points = {{3, 3}, {5, -1}, {-2, 4}};
        char[] chars = {'a', 'a', 'b', 'b', 'a', 'a'};
        String[] tokens = {"4", "13", "5", "/", "+"};
        String Str = new String("This");
        String s = "abcabcbb";
        String t = "car";
        int n = 10;
        int target = 4;
        String filePath = " "; // 替换为您的文件路径

        Trie trie = new Trie();
        trie.insert("apple");
        trie.insert("haed");
        ;
        RateLimiter rateLimiter = new RateLimiter(16, 12);
        for (int i = 0; i < nums.length; i++) {
            System.out.println(rateLimiter.shouldAllow(nums[i]));
        }

        System.out.print("返回值 :");
        //System.out.println(exculpate.longestSubarray(nums, target));

        //二维List
        List<List<Integer>> rooms = new ArrayList<List<Integer>>();
        List<Integer> room0 = new ArrayList<>();
        room0.add(1);
        List<Integer> room1 = new ArrayList<>();
        room1.add(2);
        List<Integer> room2 = new ArrayList<>();
        room2.add(3);
        List<Integer> room3 = new ArrayList<>();
        rooms.add(room0);
        rooms.add(room1);
        rooms.add(room2);
        rooms.add(room3);

//        List<String> lineList = new ArrayList<>();
        //文件读取
//        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
//            String line;
//            while ((line = br.readLine()) != null) {
//                // 对每一行内容进行处理
//                lineList.add(line);
//                // 您可以在这里添加其他处理逻辑
//            }
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

//        int[] arr = {10, 7, 8, 9, 1, 5, 1, 2, 16, 6};
//        quickSort(arr, 0, arr.length - 1);
//        System.out.println("Sorted array:");
//        printArray(arr);
    }
}


