import java.util.ArrayList;

import java.util.Scanner;
/*
 * ��С�˷�ת��9���ַ���Ϊһ�������飬��ͷ�ġ�0���͡�1����ʾ��˺�С�ˡ����������������ģʽ��
 * ��012345678������>��87654321������112345678������>��12345678��
 * ����Ϊһ�����ֺ�һ���ַ��������ֱ�ʾ�ж�������룬��Ҫ��������ӡ��������ַ��������һ�У�ÿ�����һ���ո�ָ�����β�޿ո�
 * ���磺
 * 2
 * 0abcdefgh1abcdefgh
 * �����
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