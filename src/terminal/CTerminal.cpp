#include "CTerminal.h"

void CTerminal::ShowTime(unsigned long secondes)
{
    int ss = secondes % 60;
    int mn = (secondes / 60) % 60;
    int hh = (secondes / 3600);
    printf("%2.2dh%2.2d'%2.2d", hh, mn, ss);
}

CTerminal::CTerminal(CErrorAnalyzer *_counter, CTimer *_timer, double _eb_n0){
    counter   = _counter;
    timer     = _timer;
    Eb_N0     = _eb_n0;
}


void CTerminal::temp_report(){
    if(counter->nb_be() != 0 ){
        double tBER         = counter->ber_value();
        unsigned long temps = (timer->get_time_sec() < 1) ? 1 : timer->get_time_sec();
        unsigned long fpmn  = (60 * counter->nb_processed_frames()) / temps;
        double        bps   = ((double)fpmn * (double)counter->nb_data()) / 60.0 / 1000.0 / 1000.0;
        printf("(RT) FRA: %8ld | FE: %3d | BER: %2.2e | FPM: %3ld | BPS: %2.2f | ETA: ", counter->nb_processed_frames(), (int)counter->nb_fe(), tBER, fpmn, bps);
        ShowTime( temps );
        int eta  = (int)(((double)temps / (double)counter->nb_fe()) * (counter->fe_limit()));
        printf(" | ETR: ");
        ShowTime( eta );
        printf("\r");
    }else{
        double tBER         = ( 1.0) / ((double)counter->nb_processed_frames()) / counter->nb_vars();
        unsigned long temps = (timer->get_time_sec() < 1) ? 1 : timer->get_time_sec();
        unsigned long fpmn  = (60 * counter->nb_processed_frames()) / temps;
        double        bps   = ((double)fpmn * (double)counter->nb_data()) / 60.0 / 1000.0 / 1000.0;
        printf("(RT) FRA: %8ld | FE: %3d | BER: <%2.2e | FPM: %3ld | BPS: %2.2f | ETA: ", counter->nb_processed_frames(), (int)counter->nb_fe(), tBER, fpmn, bps);
        ShowTime( temps );
        printf(" | ETR: INF.");
        printf("\r");
    }
    fflush(stdout);
}


void CTerminal::final_report(){
    double tBER = counter->ber_value();
    double tFER = counter->fer_value();
    unsigned long temps = timer->get_time_sec() + 1;
    unsigned long fpmn  = (60 * counter->nb_processed_frames()) / temps;
    double        bps   = ((double)fpmn * (double)counter->nb_data()) / 60.0 / 1000.0 / 1000.0;
    printf("SNR = %.2f | BER =  %2.3e | FER =  %2.3e | BPS =  %2.2f | MATRICES = %10ld| FE = %d | BE = %d | RUNTIME = ", Eb_N0, tBER, tFER, bps, counter->nb_processed_frames(), (int)counter->nb_fe(), (int)counter->nb_be());
    ShowTime( temps );
    printf("\n");
    fflush(stdout);
}
