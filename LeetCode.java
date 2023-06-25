import java.util.Arrays;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;

public class LeetCode {
    /**
     * 345. 反转字符串中的元音字母
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

    public static void main(String[] args) {
        LeetCode exculpate = new LeetCode();
        String s = "keycode";
        int[] nums = {1, 2, 3, 4, 5, 6, 7, 21, 3, 4, 1};
        int[] sum = new int[5];
        sum[1] = 1;
        sum[2] = 2;
        for (int index : sum) {
            System.out.println(index);

        }
    }
}
