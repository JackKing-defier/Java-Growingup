import java.util.Arrays;
import java.util.Scanner;
import java.io.*;
import java.net.UnknownHostException;
import java.util.regex.Pattern;

/**
 * @author Godfather
 *华为实习笔试题03，判断两个IP是否来自同一网段（将IP和子网掩码做与操作，输入为三个01IP字符串，以空格分开，输出0或1判断是否属于同一网段，另外输出第一个IP和子网掩码的与操作结果）
 */
public class IPtest {

    /*
     * 获取IP的地址位
     */
    private static String getAddrIp(String ip, String subnetMask) {
        StringBuilder addrIp = new StringBuilder();
        for (int i = 0; i < 32; i++) {
            byte ipByte = Byte.parseByte(String.valueOf(ip.charAt(i)));
            byte subnetMaskByte = Byte.parseByte(String.valueOf(subnetMask.charAt(i)));
            addrIp.append(ipByte & subnetMaskByte);
        }
        return addrIp.toString();
    }

    /*
     * 判断是否同一网段，指定子网掩码
     */
    public static int isSameAddress(String resourceIp, String requestIp, String subnetMask) {
        String resourceAddr = getAddrIp(resourceIp, subnetMask);
        String subnetMaskAddr = getAddrIp(requestIp, subnetMask);
        if (resourceAddr.equals(subnetMaskAddr)) {
            return 1;
        }
        return 0;
    }

    public static void main(String[] args) {
        // Scanner sc = new Scanner(System.in);
        // String inputValue = sc.nextLine();
        String inputValue = "10001000100010001000100010001000 10001000100010001000100001011100 10001000100010001000100000000000";
        String[] IPs = inputValue.split(" ");
        System.out.println(isSameAddress(IPs[0], IPs[1], IPs[2]) + " " + getAddrIp(IPs[0], IPs[2]));
    }

}
