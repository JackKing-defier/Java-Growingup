import java.util.ArrayList;

import java.util.Scanner;
/*
 * 大小端反转，9个字符串为一个编码组，开头的“0”和“1”表示大端和小端。编码解析后输出大端模式。
 * “012345678”——>“87654321”，“112345678”——>“12345678”
 * 输入为一个数字和一个字符串，数字表示有多少组编码，需要解析并打印解析后的字符串。输出一行，每个结果一个空格分隔，行尾无空格。
 * 比如：
 * 2
 * 0abcdefgh1abcdefgh
 * 输出：
 * hgfedcba abcdefgh
 * 
 */
public class Fuck {

	public static void main(String[] args) {

          // TODO Auto-generated method stub

          Scanner scan = new Scanner(System.in);

          int num = 0;

          String code = "";

          if(scan.hasNext()) {

            num = scan.nextInt();

            code = scan.next();

        }

        scan.close();

        ArrayList<String> result = new ArrayList();

        for (int i = 0; i < num; i++) {

          String substr = code.substring(i*9,i*9+9);

          if(Integer.parseInt(substr.substring(0,1)) == 0) 
{

               substr = new StringBuffer(substr.substring(1,9)).reverse().toString();

               result.add(substr);

          }else {

                result.add(substr.substring(1,9));

          }

        }

        System.out.print(result.get(0));

        for(int j = 1; j < result.size(); j++) {

          System.out.print(" " + result.get(j));

        }

     }

}